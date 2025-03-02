use chrono::DateTime;
use colored::*;
use dialoguer::{theme::ColorfulTheme, FuzzySelect, Input};
use figment::{
    providers::{Env, Format, Json},
    Figment,
};
use futures::StreamExt;
use gemini_ai_rust::{
    client::GenerativeModel,
    error::GoogleGenerativeAIError,
    models::{Content, EmbedContentRequest, ModelParams, Part, Request, TaskType},
};
use indicatif::{ProgressBar, ProgressStyle};
use pdf_extract::extract_text_by_pages;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::SystemTime,
};
use thiserror::Error;
use tokio::sync::Semaphore;

const MAX_CONCURRENT_REQUESTS: usize = 5;
const OVERLAP_PERCENTAGE: usize = 25; // Percentage of overlap between chunks
const MAX_CACHE_SIZE_MB: u64 = 200; // Maximum cache size in MB

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    google_api_key: String,
    generative_model: String,
    embedding_model: String,
}

impl Config {
    fn load() -> Result<Self, Box<dyn Error>> {
        let home_dir = dirs::home_dir().ok_or("Could not find home directory")?;
        let config_dir = home_dir.join(".gemini-ai-rust");
        std::fs::create_dir_all(&config_dir)?;

        let config_path = config_dir.join("config.json");

        // Create default config file if it doesn't exist
        if !config_path.exists() {
            std::fs::write(&config_path, r#"{ "google_api_key": "", "generative_model": "gemini-1.5-flash", "embedding_model": "embedding-001" }"#)?;
        }

        // Load config from multiple sources, with following precedence:
        // 1. Environment variables (GEMINI_GOOGLE_API_KEY)
        // 2. Config file (~/.gemini-ai-rust/config.json)
        let config: Config = Figment::new()
            .merge(Json::file(config_path))
            .merge(Env::prefixed("GEMINI_"))
            .extract()?;

        if config.google_api_key.is_empty() {
            return Err("API key not found. Please set it in ~/.gemini-ai-rust/config.json or GEMINI_GOOGLE_API_KEY environment variable".into());
        }

        Ok(config)
    }
}

// Custom chat session implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    role: String,
    content: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    history: Vec<Message>,
    max_messages: usize,
}

impl ChatSession {
    pub fn new(max_messages: usize) -> Self {
        Self {
            history: Vec::new(),
            max_messages,
        }
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        let message = Message::new(role, content);
        self.history.push(message);

        // Maintain max message limit
        if self.history.len() > self.max_messages {
            self.history.remove(0);
        }
    }

    pub fn get_formatted_history(&self) -> String {
        self.history
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}

// Custom error types
#[derive(Error, Debug)]
pub enum ChatError {
    #[error("Failed to generate embedding: {0}")]
    EmbeddingGeneration(String),
    #[error("No content found")]
    NoContent,
    #[error("API error: {0}")]
    ApiError(#[from] GoogleGenerativeAIError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("PDF error: {0}")]
    PdfError(#[from] pdf_extract::OutputError),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

// Text chunk with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    content: String,
    section: usize,
    page: usize,
    embedding: Option<Vec<f32>>,
}

// Cache structure
#[derive(Debug, Serialize, Deserialize)]
struct DocumentCache {
    text: String,
    chunks: Vec<TextChunk>,
    last_modified: chrono::DateTime<chrono::Utc>,
}

// File structure
#[derive(Debug, Clone)]
pub struct EntityFile {
    pub index: usize,
    pub path: PathBuf,
    pub name: String,
    pub size: u64,
    pub last_modified: SystemTime,
}

impl EntityFile {
    pub fn size_formatted(&self) -> String {
        const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let mut size = self.size as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    fn truncate_name(&self, max_length: usize) -> String {
        if self.name.len() > max_length {
            format!("{}…", &self.name[..max_length - 1])
        } else {
            self.name.clone()
        }
    }

    fn format_date(&self) -> String {
        self.last_modified
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .and_then(|d| DateTime::from_timestamp(d.as_secs() as i64, 0))
            .map_or_else(
                || "Unknown".to_string(),
                |dt| dt.format("%Y-%m-%d %H:%M").to_string(),
            )
    }
}

impl std::fmt::Display for EntityFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{index:>3} │ {name:<35} │ {size:>10} │ {date}",
            index = self.index.to_string().bright_yellow(),
            name = self.truncate_name(35).bright_green(),
            size = self.size_formatted().bright_blue(),
            date = self.format_date().bright_black(),
        )
    }
}

impl AsRef<Path> for EntityFile {
    fn as_ref(&self) -> &Path {
        self.path.as_ref()
    }
}

/// Get a list of files from a directory
///
/// # Arguments
/// * `dir` - The directory to search in
/// * `extension` - The file extension to filter by
/// * `min_size` - Optional minimum file size in bytes
///
/// # Returns
/// * `Result<Vec<EntityFile>, std::io::Error>` - A list of EntityFile structs
pub async fn get_files(
    dir: impl AsRef<std::path::Path>,
    extension: &str,
    min_size: Option<u64>,
) -> Result<Vec<EntityFile>, std::io::Error> {
    let mut matching_files = Vec::new();
    let mut entries = tokio::fs::read_dir(dir).await?;

    let mut index = 0;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case(extension))
        {
            let metadata = entry.metadata().await?;
            let size = metadata.len();
            if min_size.map_or(true, |min| size >= min) {
                index += 1;
                matching_files.push(EntityFile {
                    index,
                    name: path.file_name().unwrap().to_string_lossy().into_owned(),
                    path,
                    size,
                    last_modified: metadata.modified().unwrap(),
                });
            }
        }
    }

    Ok(matching_files)
}

// Document chat manager
pub struct DocumentChatManager {
    model: GenerativeModel,
    model_name: String,
    chunks: Vec<TextChunk>,
    chat_session: Option<ChatSession>,
    chunk_size: usize,
    cache_dir: PathBuf,
    current_pdf_path: Option<PathBuf>,
    semaphore: Arc<Semaphore>,
}

impl DocumentChatManager {
    pub fn new(model: GenerativeModel, model_name: impl Into<String>, chunk_size: usize) -> Self {
        let home_dir = dirs::home_dir().expect("Failed to get home directory");
        let cache_dir = home_dir.join(".cache").join("pdf_chat");
        fs::create_dir_all(&cache_dir).unwrap_or_else(|_| {
            eprintln!("Warning: Failed to create cache directory");
        });

        Self {
            model,
            model_name: model_name.into(),
            chunks: Vec::new(),
            chat_session: None,
            chunk_size,
            cache_dir,
            current_pdf_path: None,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
        }
    }

    fn get_cache_path(&self, pdf_path: &Path) -> PathBuf {
        // Create a unique cache file name based on the PDF path and last modified time
        let last_modified = fs::metadata(pdf_path)
            .and_then(|m| m.modified())
            .unwrap_or_else(|_| std::time::SystemTime::now());

        let mut hasher = Sha256::new();
        hasher.update(pdf_path.to_string_lossy().as_bytes());
        hasher.update(format!("{:?}", last_modified).as_bytes());
        let hash = hex::encode(hasher.finalize());

        self.cache_dir.join(format!("{}.json", &hash[..16]))
    }

    fn load_cache(&self, pdf_path: &Path) -> Result<Option<DocumentCache>, ChatError> {
        let cache_path = self.get_cache_path(pdf_path);
        if !cache_path.exists() {
            return Ok(None);
        }

        let mut file = File::open(cache_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let cache: DocumentCache = serde_json::from_str(&contents)?;

        // Verify if cache is still valid
        let pdf_modified = fs::metadata(pdf_path)?.modified()?;
        let cache_time: SystemTime = cache
            .last_modified
            .naive_utc()
            .and_local_timezone(chrono::Utc)
            .unwrap()
            .into();

        if pdf_modified > cache_time {
            return Ok(None);
        }

        Ok(Some(cache))
    }

    fn save_cache(&self, pdf_path: impl AsRef<Path>, text: &str) -> Result<(), ChatError> {
        let cache = DocumentCache {
            text: text.to_string(),
            chunks: self.chunks.clone(),
            last_modified: chrono::Utc::now(),
        };

        let cache_path = self.get_cache_path(pdf_path.as_ref());
        let file = File::create(cache_path)?;
        serde_json::to_writer_pretty(file, &cache)?;
        Ok(())
    }

    fn manage_cache_size(&self) -> Result<(), ChatError> {
        let mut cache_files: Vec<_> = fs::read_dir(&self.cache_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| (entry.path(), entry.metadata().unwrap().modified().unwrap()))
            .collect();

        // Sort by last modified time (oldest first)
        cache_files.sort_by_key(|&(_, time)| time);

        let mut total_size = 0u64;
        for (path, _) in &cache_files {
            total_size += fs::metadata(path)?.len();
        }

        // Remove oldest files if total size exceeds limit
        let max_size = MAX_CACHE_SIZE_MB * 1024 * 1024;
        if total_size > max_size {
            for (path, _) in cache_files {
                if total_size <= max_size {
                    break;
                }
                if let Ok(size) = fs::metadata(&path).map(|m| m.len()) {
                    fs::remove_file(path)?;
                    total_size -= size;
                }
            }
        }

        Ok(())
    }

    fn process_text_chunk(text: &str, chunk_size: usize, overlap: usize) -> Vec<TextChunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut current_section = 1;
        let mut i = 0;

        while i < words.len() {
            let end = (i + chunk_size).min(words.len());
            let chunk = words[i..end].join(" ");

            chunks.push(TextChunk {
                content: chunk,
                section: current_section,
                page: 0,
                embedding: None,
            });

            i += chunk_size - overlap;
            if i >= words.len() {
                break;
            }

            if chunks.len() % 5 == 0 {
                current_section += 1;
            }
        }

        chunks
    }

    pub fn process_text(&mut self, text: &str) -> Result<(), ChatError> {
        if text.is_empty() {
            return Err(ChatError::NoContent);
        }

        let overlap = (self.chunk_size * OVERLAP_PERCENTAGE) / 100;
        self.chunks = Self::process_text_chunk(text, self.chunk_size, overlap);
        Ok(())
    }

    pub async fn generate_embeddings(&mut self) -> Result<(), ChatError> {
        let pb = ProgressBar::new(self.chunks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let chunks = Arc::new(Mutex::new(self.chunks.clone()));
        let mut handles = Vec::new();
        let semaphore = self.semaphore.clone();

        for i in 0..self.chunks.len() {
            if self.chunks[i].embedding.is_some() {
                pb.inc(1);
                continue;
            }

            let chunks = Arc::clone(&chunks);
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let model = self.model.clone();
            let model_name = self.model_name.clone();
            let pb = pb.clone();

            let handle = tokio::spawn(async move {
                let chunk = {
                    let chunks = chunks.lock().unwrap();
                    chunks[i].content.clone()
                };

                let request =
                    EmbedContentRequest::new(&chunk, Some(TaskType::RetrievalDocument), None);
                let result = model.embed_content(&model_name, request).await;

                if let Ok(response) = result {
                    let mut chunks = chunks.lock().unwrap();
                    chunks[i].embedding = Some(response.embedding.values);
                }

                pb.inc(1);
                drop(permit);
            });

            handles.push(handle);
        }

        // Wait for all embeddings to complete
        for handle in handles {
            handle
                .await
                .map_err(|e| ChatError::EmbeddingGeneration(e.to_string()))?;
        }

        pb.finish_with_message("Embeddings generated");

        // Update chunks with the generated embeddings
        self.chunks = Arc::try_unwrap(chunks).unwrap().into_inner().unwrap();

        // Update cache if needed
        if let Some(path) = self.current_pdf_path.as_ref() {
            self.save_cache(path, "")?;
        }

        Ok(())
    }

    fn extract_text_from_pages(pdf_path: &Path) -> Result<Vec<PageInfo>, ChatError> {
        extract_text_by_pages(pdf_path)?
            .into_iter()
            .enumerate()
            .map(|(idx, page_text)| {
                Ok(PageInfo {
                    page_number: idx + 1,
                    content: page_text.clone(),
                    word_count: page_text.split_whitespace().count(),
                })
            })
            .collect()
    }

    pub fn process_pdf(&mut self, path: impl AsRef<Path>) -> Result<(), ChatError> {
        self.manage_cache_size()?;
        self.current_pdf_path = Some(path.as_ref().to_path_buf());

        // Try to load from cache first
        if let Some(cache) = self.load_cache(path.as_ref())? {
            self.chunks = cache.chunks;
            return Ok(());
        }

        // Extract text by pages
        let pages = Self::extract_text_from_pages(path.as_ref())?;

        // Process each page
        let mut current_section = 1;
        let mut chunks = Vec::new();

        for page in pages {
            let text = page.content;
            let words: Vec<&str> = text.split_whitespace().collect();
            let overlap = (self.chunk_size * OVERLAP_PERCENTAGE) / 100;

            let mut i = 0;
            while i < words.len() {
                let end = (i + self.chunk_size).min(words.len());
                let chunk = words[i..end].join(" ");

                chunks.push(TextChunk {
                    content: chunk,
                    section: current_section,
                    page: page.page_number,
                    embedding: None,
                });

                i += self.chunk_size - overlap;
                if chunks.len() % 5 == 0 {
                    current_section += 1;
                }
            }
        }

        self.chunks = chunks;
        self.save_cache(path, "")?;
        Ok(())
    }

    pub async fn find_relevant_chunks(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<&TextChunk>, ChatError> {
        let request = EmbedContentRequest::new(query, Some(TaskType::RetrievalQuery), None);
        let response = self.model.embed_content(&self.model_name, request).await?;
        let query_embedding = response.embedding.values;

        // Use rayon for parallel processing of similarity calculations
        let chunks_with_scores: Vec<_> = self
            .chunks
            .par_iter()
            .filter_map(|chunk| {
                chunk.embedding.as_ref().map(|emb| {
                    let similarity = calculate_similarity(&query_embedding, emb);
                    (chunk, similarity)
                })
            })
            .collect();

        // Sort results (must be done sequentially)
        let mut chunks_with_scores = chunks_with_scores;
        chunks_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(chunks_with_scores
            .into_iter()
            .take(limit)
            .map(|(chunk, _)| chunk)
            .collect())
    }

    pub fn start_chat_session(&mut self) {
        self.chat_session = Some(ChatSession::new(10)); // Keep last 10 messages
    }

    pub async fn chat(&mut self, user_input: &str) -> Result<(), ChatError> {
        if self.chat_session.is_none() {
            self.start_chat_session();
        }

        let relevant_chunks = self.find_relevant_chunks(user_input, 3).await?;
        let context = relevant_chunks
            .iter()
            .map(|chunk| {
                format!(
                    "Content from page {} (section {}): {}",
                    chunk.page, chunk.section, chunk.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let chat_session = self.chat_session.as_mut().unwrap();
        chat_session.add_message("user", user_input);

        let prompt = format!(
            "Use the following context to answer the user's question.\n\n\
             Context from the document:\n{}\n\n\
             Chat history:\n{}\n\n",
            context,
            chat_session.get_formatted_history()
        );

        let request = Request::builder()
            .system_instruction(Some(
                "You are a helpful assistant analyzing a document. \
            Please provide a clear and concise answer based on the context. \
            When referring to content, mention the page number where it appears. \
            Carefully heed the user's instructions. \
            Respond using Markdown."
                    .into(),
            ))
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: prompt }],
            }])
            .build();

        let mut stream = self.model.stream_generate_response(request).await?;
        let mut accumulated_response = String::new();
        let mut first_chunk = true;

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(response) => {
                    if !first_chunk {
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                    let text = response.text();
                    accumulated_response.push_str(&text);
                    PrettyPrinter::print_streaming_message(&text, true);
                    first_chunk = false;
                }
                Err(e) => {
                    PrettyPrinter::print_error(&e);
                    break;
                }
            }
        }

        // Add the complete response to chat history
        if !accumulated_response.is_empty() {
            chat_session.add_message("assistant", &accumulated_response);
        }

        Ok(())
    }

    pub fn get_stats(&self) -> Result<DocumentStats, ChatError> {
        let file_size = if let Some(path) = &self.current_pdf_path {
            fs::metadata(path)?.len()
        } else {
            0
        };

        let total_words = self
            .chunks
            .iter()
            .map(|chunk| chunk.content.split_whitespace().count())
            .sum();

        Ok(DocumentStats {
            total_words,
            total_sections: self.chunks.last().map(|c| c.section).unwrap_or(0),
            total_chunks: self.chunks.len(),
            processed_date: chrono::Utc::now(),
            file_size,
        })
    }

    pub fn list_sections(&self) -> Vec<String> {
        let mut sections = Vec::new();
        let mut current_section = 0;
        let mut current_content = String::new();

        for chunk in &self.chunks {
            if chunk.section != current_section {
                if !current_content.is_empty() {
                    let preview = if current_content.len() > 100 {
                        format!("{}...", &current_content[..100])
                    } else {
                        current_content.clone()
                    };
                    sections.push(format!("Section {}: {}", current_section, preview));
                }
                current_section = chunk.section;
                current_content = chunk.content.clone();
            } else {
                current_content.push_str(&chunk.content);
            }
        }

        sections
    }

    pub async fn generate_summary(&self) -> Result<String, ChatError> {
        let sections = self.list_sections();
        let context = sections.join("\n\n");

        let prompt = format!(
            "Please provide a concise summary of this document. Focus on the main topics and key points.\n\nDocument content:\n{}",
            context
        );

        let request = Request::builder()
            .system_instruction(Some(
                "You are a helpful assistant analyzing a document. \
            Carefully heed the user's instructions. \
            Respond using Markdown."
                    .into(),
            ))
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: prompt }],
            }])
            .build();

        let response = self.model.generate_response(request).await?;
        Ok(response.text())
    }

    pub fn export_chat(&self, filename: &str) -> Result<(), ChatError> {
        if let Some(chat_session) = &self.chat_session {
            let export = ChatExport {
                document_name: self
                    .current_pdf_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default(),
                chat_history: chat_session.history.clone(),
                export_date: chrono::Utc::now(),
                document_stats: self.get_stats()?,
            };

            let file = File::create(filename)?;
            serde_json::to_writer_pretty(file, &export)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStats {
    total_words: usize,
    total_sections: usize,
    total_chunks: usize,
    processed_date: chrono::DateTime<chrono::Utc>,
    file_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatExport {
    document_name: String,
    chat_history: Vec<Message>,
    export_date: chrono::DateTime<chrono::Utc>,
    document_stats: DocumentStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PageInfo {
    page_number: usize,
    content: String,
    word_count: usize,
}

fn calculate_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

const BANNER: &str = r#"
╔═══════════════════════════════════════════════════════════════════════════╗
║                     🤖 PDF Chat Assistant Pro v1.0                         ║
║                     Powered by Gemini AI & Rust                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"#;

const HELP_TEXT: &str = r#"
Available Commands:
  /help     - Show this help message
  /clear    - Clear chat history
  /stats    - Show document statistics
  /sections - List document sections
  /export   - Export chat history to file
  /summary  - Generate document summary
  /exit     - Exit the application
"#;

pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print_banner() {
        println!("{}", BANNER.bright_cyan());
    }

    pub fn print_help() {
        println!("{}", HELP_TEXT.bright_yellow());
    }

    pub fn print_stats(stats: &DocumentStats) {
        println!("\n{}", "📊 Document Statistics".bright_cyan().bold());
        println!("{}", "═".repeat(50).bright_black());
        println!(
            "📝 Total Words: {}",
            stats.total_words.to_string().bright_green()
        );
        println!(
            "📑 Sections: {}",
            stats.total_sections.to_string().bright_green()
        );
        println!(
            "🧩 Chunks: {}",
            stats.total_chunks.to_string().bright_green()
        );
        println!(
            "📦 File Size: {} MB",
            (stats.file_size as f64 / 1_048_576.0)
                .round()
                .to_string()
                .bright_green()
        );
        println!(
            "🕒 Processed: {}",
            stats
                .processed_date
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
                .bright_green()
        );
    }

    pub fn print_sections(sections: &[String]) {
        println!("\n{}", "📑 Document Sections".bright_cyan().bold());
        println!("{}", "═".repeat(50).bright_black());
        for section in sections {
            println!("{}", section.bright_green());
        }
    }

    pub fn print_chat_message(role: &str, message: &str) {
        let (prefix, color) = match role {
            "user" => ("You:", "blue"),
            "model" => ("Model:", "green"),
            "system" => ("System:", "yellow"),
            _ => ("Unknown:", "red"),
        };

        println!("\n{}", "─".repeat(100).bright_black());
        match color {
            "blue" => println!("{} {}", prefix.blue().bold(), message),
            "green" => println!("{} {}", prefix.green().bold(), message),
            "yellow" => println!("{} {}", prefix.yellow().bold(), message),
            _ => println!("{} {}", prefix.red().bold(), message),
        }
        println!("{}", "─".repeat(100).bright_black());
    }

    pub fn print_streaming_message(text: &str, page_references: bool) {
        let text = text.replace("**", "\x1b[1m").replace("`", "\x1b[36m");
        if page_references {
            let re = Regex::new(r"([Pp]age\s+\d+)").unwrap();
            let text = re.replace_all(&text, |caps: &regex::Captures| {
                format!("{}", caps[1].bright_cyan().bold())
            });
            print!("{}\x1b[0m", text.white());
        } else {
            print!("{}\x1b[0m", text.white());
        }
        std::io::stdout().flush().unwrap();
    }

    pub fn print_success(message: &str) {
        println!("\n✓ {}", message.green());
    }

    pub fn print_error(error: &dyn Error) {
        eprintln!("\n{} {}", "Error:".red().bold(), error.to_string().red());
    }

    pub fn print_thinking() -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.set_message("Thinking...");
        pb
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\n\n{}", "👋 Gracefully shutting down...".bright_yellow());
        std::process::exit(0);
    })?;

    PrettyPrinter::print_banner();
    PrettyPrinter::print_help();

    let pdf_files = get_files(std::env::current_dir()?, "pdf", None).await?;

    if pdf_files.is_empty() {
        println!(
            "{}",
            "❌ No PDF files found in the current directory!".bright_red()
        );
        return Ok(());
    }

    // Use dialoguer to select a PDF file
    let selection = FuzzySelect::new()
        .with_prompt("Select a PDF file")
        .items(&pdf_files)
        .interact()?;

    let file = &pdf_files[selection];

    println!("\n{}", "🔄 Initializing...".bright_blue());

    // Load configuration
    let config = Config::load()?;

    let model = GenerativeModel::new(
        config.google_api_key.clone(),
        ModelParams::builder().model(&config.generative_model).build(),
    );
    let mut doc_manager = DocumentChatManager::new(model.clone(), &config.embedding_model, 200);
    PrettyPrinter::print_success("Document manager initialized");

    println!("\n{}", "📄 Processing PDF...".bright_blue());
    match doc_manager.process_pdf(&file.path) {
        Ok(_) => PrettyPrinter::print_success("PDF processed successfully"),
        Err(e) => {
            PrettyPrinter::print_error(&e);
            return Ok(());
        }
    }

    println!("\n{}", "🧠 Generating embeddings...".bright_blue());
    match doc_manager.generate_embeddings().await {
        Ok(_) => PrettyPrinter::print_success("Embeddings generated successfully"),
        Err(e) => {
            PrettyPrinter::print_error(&e);
            return Ok(());
        }
    }

    doc_manager.start_chat_session();
    PrettyPrinter::print_success("Chat session started");

    println!(
        "\n{}",
        "🚀 Ready to chat! Type /help for available commands.".bright_green()
    );
    println!("{}", "═".repeat(50).bright_black());

    loop {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        print!("\n{} ", "You:".blue().bold());
        std::io::stdout().flush()?;

        // Use dialoguer for user input
        let input: String = Input::with_theme(&ColorfulTheme::default())
            .allow_empty(true) // Allow empty input
            .interact_text()?;

        let input = input.trim();

        match input {
            "/help" => PrettyPrinter::print_help(),
            "/clear" => {
                doc_manager.chat_session = Some(ChatSession::new(10));
                PrettyPrinter::print_success("Chat history cleared");
            }
            "/stats" => {
                if let Ok(stats) = doc_manager.get_stats() {
                    PrettyPrinter::print_stats(&stats);
                }
            }
            "/sections" => {
                let sections = doc_manager.list_sections();
                PrettyPrinter::print_sections(&sections);
            }
            "/export" => {
                let filename = format!(
                    "chat_export_{}.json",
                    chrono::Utc::now().format("%Y%m%d_%H%M%S")
                );
                if doc_manager.export_chat(&filename).is_ok() {
                    PrettyPrinter::print_success(&format!("Chat exported to {}", filename));
                }
            }
            "/summary" => {
                let pb = PrettyPrinter::print_thinking();
                match doc_manager.generate_summary().await {
                    Ok(summary) => {
                        pb.finish_and_clear();
                        PrettyPrinter::print_chat_message("system", &summary);
                    }
                    Err(e) => PrettyPrinter::print_error(&e),
                }
            }
            "/exit" => {
                println!("\n{}", "👋 Goodbye!".bright_yellow());
                break;
            }
            _ => {
                if input.is_empty() {
                    println!("{}", "❗ Please enter a question".bright_yellow());
                    continue;
                }

                let pb = PrettyPrinter::print_thinking();
                println!("\n{}", "─".repeat(80).bright_black());
                println!("{}", "🤖 Response:".green().bold());
                println!("{}", "─".repeat(80).bright_black());

                match doc_manager.chat(input).await {
                    Ok(_) => println!("\n{}", "─".repeat(80).bright_black()),
                    Err(e) => {
                        pb.finish_and_clear();
                        PrettyPrinter::print_error(&e);
                    }
                }
            }
        }
    }

    Ok(())
}
