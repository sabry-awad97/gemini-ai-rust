use base64::{engine::general_purpose::STANDARD as base64_engine, Engine};
use colored::*;
use dialoguer::Input;
use figment::{
    providers::{Env, Format, Json},
    Figment,
};
use gemini_ai_rust::{
    client::GenerativeModel,
    error::GoogleGenerativeAIError,
    file::{FileError, FileState, GoogleAIFileManager},
    models::{Content, InlineData, ModelParams, Part, Request},
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};
use thiserror::Error;

const MAX_CACHE_SIZE_MB: u64 = 1000; // 1 GB
const CACHE_DIR: &str = ".file_chat_cache";

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

        if !config_path.exists() {
            std::fs::write(
                &config_path,
                r#"{ "google_api_key": "", "generative_model": "gemini-1.5-flash", "embedding_model": "embedding-001" }"#,
            )?;
        }

        let config: Config = Figment::new()
            .merge(Json::file(config_path))
            .merge(Env::prefixed("GEMINI_"))
            .extract()?;

        if config.google_api_key.is_empty() {
            return Err(
                "API key not found. Please set it in ~/.gemini-ai-rust/config.json or GEMINI_GOOGLE_API_KEY environment variable"
                    .into(),
            );
        }

        Ok(config)
    }
}

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
    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),
    #[error("File too large: {0}")]
    FileTooLarge(String),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("File processing error: {0}")]
    FileProcessing(String),
    #[error("File management error: {0}")]
    FileManagement(#[from] FileError),
}

#[derive(Debug, Serialize, Deserialize)]
struct FileCache {
    hash: String,
    mime_type: String,
    content: Vec<u8>,
    last_accessed: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
pub struct FileChatManager {
    model: GenerativeModel,
    chat_session: ChatSession,
    current_file: Option<PathBuf>,
    file_content: Option<Vec<u8>>,
    mime_type: Option<String>,
    cache_dir: PathBuf,
    file_manager: GoogleAIFileManager,
    current_file_info: Option<String>, // Stores the Google AI file name
}

impl FileChatManager {
    pub fn new(model: GenerativeModel, api_key: &str) -> Result<Self, ChatError> {
        let home_dir = dirs::home_dir().ok_or_else(|| {
            ChatError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find home directory",
            ))
        })?;
        let cache_dir = home_dir.join(".gemini-ai-rust").join(CACHE_DIR);
        std::fs::create_dir_all(&cache_dir)?;

        let file_manager = GoogleAIFileManager::new(api_key);

        Ok(Self {
            model,
            chat_session: ChatSession::new(50),
            current_file: None,
            file_content: None,
            mime_type: None,
            cache_dir,
            file_manager,
            current_file_info: None,
        })
    }

    fn calculate_file_hash(content: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content);
        format!("{:x}", hasher.finalize())
    }

    fn get_cache_path(&self, hash: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.cache", hash))
    }

    fn load_from_cache(&self, hash: &str) -> Result<Option<FileCache>, ChatError> {
        let cache_path = self.get_cache_path(hash);
        if !cache_path.exists() {
            return Ok(None);
        }

        let cache_file = File::open(cache_path)?;
        let cache: FileCache = serde_json::from_reader(cache_file)?;
        Ok(Some(cache))
    }

    fn save_to_cache(&self, cache: &FileCache) -> Result<(), ChatError> {
        let cache_path = self.get_cache_path(&cache.hash);
        let cache_file = File::create(cache_path)?;
        serde_json::to_writer(cache_file, cache)?;
        self.cleanup_old_cache()?;
        Ok(())
    }

    fn cleanup_old_cache(&self) -> Result<(), ChatError> {
        let max_cache_size = MAX_CACHE_SIZE_MB * 1024 * 1024;
        let mut cache_files: Vec<_> = fs::read_dir(&self.cache_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "cache"))
            .collect();

        if cache_files.is_empty() {
            return Ok(());
        }

        // Sort by last modified time
        cache_files.sort_by_key(|entry| {
            entry
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });

        let mut total_size: u64 = 0;
        for entry in cache_files.iter() {
            if let Ok(size) = entry.metadata().map(|m| m.len()) {
                total_size += size;
            }
        }

        // Remove oldest files until we're under the limit
        while total_size > max_cache_size && !cache_files.is_empty() {
            if let Some(oldest) = cache_files.first() {
                if let Ok(size) = oldest.metadata().map(|m| m.len()) {
                    total_size -= size;
                    fs::remove_file(oldest.path())?;
                }
                cache_files.remove(0);
            }
        }

        Ok(())
    }

    pub async fn load_file(&mut self, path: impl AsRef<Path>) -> Result<(), ChatError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;

        if metadata.len() > 20 * 1024 * 1024 {
            return Err(ChatError::FileTooLarge(format!(
                "File size {} MB exceeds limit of 20 MB",
                metadata.len() / (1024 * 1024)
            )));
        }

        let content = fs::read(path)?;
        let hash = Self::calculate_file_hash(&content);
        let mime_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        // Try to load from cache first
        if let Some(mut cache) = self.load_from_cache(&hash)? {
            // Update last accessed time
            cache.last_accessed = chrono::Utc::now();
            self.save_to_cache(&cache)?;

            self.current_file = Some(path.to_path_buf());
            self.file_content = Some(cache.content);
            self.mime_type = Some(cache.mime_type);

            println!("{}", "📦 Loaded from cache".bright_green().bold());
        } else {
            // If not in cache, save to cache
            let cache = FileCache {
                hash,
                mime_type: mime_type.clone(),
                content: content.clone(),
                last_accessed: chrono::Utc::now(),
            };
            self.save_to_cache(&cache)?;

            self.current_file = Some(path.to_path_buf());
            self.file_content = Some(content);
            self.mime_type = Some(mime_type.clone());

            println!("{}", "💾 File cached for future use".bright_green().bold());
        }

        // Upload to Google AI
        let file_name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
            ChatError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid file name",
            ))
        })?;

        println!("{}", "📤 Uploading to Google AI...".bright_yellow().bold());
        let pb = PrettyPrinter::print_thinking();

        // Upload the file
        let file_info = self
            .file_manager
            .upload_file(path, file_name.to_string())
            .await?;

        // Wait for processing if needed
        if matches!(file_info.state, FileState::Processing) {
            pb.set_message("Processing file...");
            let processed_file = self
                .file_manager
                .wait_for_file_processing(&file_info.name, 10, 1000)
                .await?;
            self.current_file_info = Some(processed_file.name);
        } else {
            self.current_file_info = Some(file_info.name);
        }

        pb.finish_and_clear();
        println!("{}", "✨ File ready for chat".bright_green().bold());

        Ok(())
    }

    pub async fn chat(&mut self, user_input: &str) -> Result<(), ChatError> {
        if self.file_content.is_none() {
            return Err(ChatError::NoContent);
        }

        let mime_type = self.mime_type.as_ref().unwrap();
        self.chat_session.add_message("user", user_input);

        let mut parts = vec![Part::Text {
            text: format!(
                "Context: I'm looking at a file of type {}. Here's my question: {}",
                mime_type, user_input
            ),
        }];

        // Add file content based on whether we have a Google AI file or local file
        if let Some(file_name) = &self.current_file_info {
            // Use Google AI file reference
            parts.push(Part::file_data(
                mime_type.clone(),
                format!("files/{}", file_name),
            ));
        } else {
            // Fallback to direct content
            if mime_type.starts_with("text/") {
                if let Ok(text) = String::from_utf8(self.file_content.as_ref().unwrap().clone()) {
                    parts.push(Part::Text { text });
                }
            } else if mime_type.starts_with("image/") {
                parts.push(Part::InlineData {
                    inline_data: InlineData {
                        mime_type: mime_type.clone(),
                        data: base64_engine.encode(self.file_content.as_ref().unwrap()),
                    },
                });
            }
        }

        let request = Request::builder()
            .contents(vec![Content { role: None, parts }])
            .build();

        match self.model.generate_response(request).await {
            Ok(response) => {
                let response_text = response.text();
                self.chat_session.add_message("model", &response_text);
                PrettyPrinter::print_chat_message("Model", &response_text);
                Ok(())
            }
            Err(e) => Err(ChatError::ApiError(e)),
        }
    }

    pub async fn cleanup(&self) -> Result<(), ChatError> {
        if let Some(file_name) = &self.current_file_info {
            println!(
                "{}",
                "🗑️  Cleaning up remote file...".bright_yellow().bold()
            );
            self.file_manager.delete_file(file_name).await?;
            println!("{}", "✨ Cleanup complete".bright_green().bold());
        }
        Ok(())
    }

    pub fn get_file_info(&self) -> Option<FileInfo> {
        if let (Some(path), Some(content)) = (&self.current_file, &self.file_content) {
            Some(FileInfo {
                path: path.clone(),
                size: content.len() as u64,
                mime_type: self
                    .mime_type
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    path: PathBuf,
    size: u64,
    mime_type: String,
}

impl FileInfo {
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
}

pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print_banner() {
        println!(
            "{}",
            r#"
╔═══════════════════════════════════════════════════════════════════════════╗
║                    🤖 File Chat Assistant Pro v1.0                         ║
║                    Powered by Gemini AI & Rust                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
"#
            .bright_cyan()
        );
    }

    pub fn print_help() {
        println!(
            "{}",
            r#"
Available Commands:
/help     - Show this help message
/load     - Load a new file
/info     - Show current file information
/clear    - Clear chat history
/exit     - Exit the program
"#
            .bright_yellow()
        );
    }

    pub fn print_file_info(info: &FileInfo) {
        println!("\n{}", "📄 File Information".bright_blue().bold());
        println!("{}", "═".repeat(50).bright_blue());
        println!(
            "{} {}",
            "Path:".yellow().bold(),
            info.path.display().to_string().bright_yellow()
        );
        println!(
            "{} {}",
            "Size:".yellow().bold(),
            info.size_formatted().bright_yellow()
        );
        println!(
            "{} {}",
            "Type:".yellow().bold(),
            info.mime_type.bright_yellow()
        );
    }

    pub fn print_chat_message(role: &str, message: &str) {
        let prefix = match role.to_lowercase().as_str() {
            "user" => "👤 You:".blue().bold(),
            "model" => "🤖 Model:".green().bold(),
            _ => role.white().bold(),
        };

        println!("\n{}", prefix);
        println!("{}", message.white());
    }

    pub fn print_error(error: &dyn Error) {
        println!("{} {}", "❌ Error:".red().bold(), error.to_string().red());
    }

    pub fn print_thinking() -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈")
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Thinking...");
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    PrettyPrinter::print_banner();
    println!(
        "{}",
        "Welcome to File Chat! Type /help for available commands."
            .bright_black()
            .italic()
    );

    let config = Config::load()?;
    let model = GenerativeModel::new(
        &config.google_api_key,
        ModelParams::builder()
            .model(&config.generative_model)
            .build(),
    );
    let mut chat_manager = FileChatManager::new(model, &config.google_api_key)?;

    loop {
        let input: String = Input::<String>::new().with_prompt("You").interact_text()?;

        match input.trim() {
            "/help" => PrettyPrinter::print_help(),
            "/exit" => {
                chat_manager.cleanup().await?;
                break;
            }
            "/load" => {
                let file_path: String = Input::new()
                    .with_prompt("Enter file path")
                    .interact_text()?;

                match chat_manager.load_file(file_path).await {
                    Ok(()) => {
                        if let Some(info) = chat_manager.get_file_info() {
                            PrettyPrinter::print_file_info(&info);
                        }
                    }
                    Err(e) => PrettyPrinter::print_error(&e),
                }
            }
            "/info" => {
                if let Some(info) = chat_manager.get_file_info() {
                    PrettyPrinter::print_file_info(&info);
                } else {
                    println!("No file loaded");
                }
            }
            "/clear" => {
                chat_manager.chat_session.clear();
                println!("Chat history cleared");
            }
            _ => {
                let pb = PrettyPrinter::print_thinking();
                match chat_manager.chat(&input).await {
                    Ok(()) => pb.finish_and_clear(),
                    Err(e) => {
                        pb.finish_and_clear();
                        PrettyPrinter::print_error(&e);
                    }
                }
            }
        }
    }

    println!("\n✨ Thanks for using File Chat! Goodbye!");
    Ok(())
}
