use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    error::GoogleGenerativeAIError,
    models::{EmbedContentRequest, TaskType},
};
use pdf_extract::extract_text;
use serde::{Deserialize, Serialize};
use std::{error::Error, io::Write};
use thiserror::Error;

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
}

// Text chunk with metadata
#[derive(Debug, Clone)]
struct TextChunk {
    content: String,
    section: usize,
    embedding: Option<Vec<f32>>,
}

// Document chat manager
pub struct DocumentChatManager {
    model: GenerativeModel,
    model_name: String,
    chunks: Vec<TextChunk>,
    chat_session: Option<ChatSession>,
    chunk_size: usize,
}

impl DocumentChatManager {
    pub fn new(model: GenerativeModel, model_name: impl Into<String>, chunk_size: usize) -> Self {
        Self {
            model,
            model_name: model_name.into(),
            chunks: Vec::new(),
            chat_session: None,
            chunk_size,
        }
    }

    pub fn process_text(&mut self, text: &str) -> Result<(), ChatError> {
        if text.is_empty() {
            return Err(ChatError::NoContent);
        }

        // Split text into chunks with overlap
        let overlap = self.chunk_size / 4; // 25% overlap
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_section = 1;

        let mut i = 0;
        while i < words.len() {
            let end = (i + self.chunk_size).min(words.len());
            let chunk = words[i..end].join(" ");

            self.chunks.push(TextChunk {
                content: chunk,
                section: current_section,
                embedding: None,
            });

            i += self.chunk_size - overlap;
            if i > words.len() {
                break;
            }

            if i % 500 == 0 {
                // New section every ~500 words
                current_section += 1;
            }
        }

        Ok(())
    }

    pub async fn generate_embeddings(&mut self) -> Result<(), ChatError> {
        for chunk in &mut self.chunks {
            let request =
                EmbedContentRequest::new(&chunk.content, Some(TaskType::RetrievalDocument), None);
            let response = self.model.embed_content(&self.model_name, request).await?;
            chunk.embedding = Some(response.embedding.values);
        }
        Ok(())
    }

    async fn find_relevant_chunks(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<&TextChunk>, ChatError> {
        let request = EmbedContentRequest::new(query, Some(TaskType::RetrievalQuery), None);
        let response = self.model.embed_content(&self.model_name, request).await?;
        let query_embedding = response.embedding.values;

        let mut chunks_with_scores: Vec<(&TextChunk, f32)> = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                chunk.embedding.as_ref().map(|emb| {
                    let similarity = calculate_similarity(&query_embedding, emb);
                    (chunk, similarity)
                })
            })
            .collect();

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

    pub async fn chat(&mut self, user_input: &str) -> Result<String, ChatError> {
        if self.chat_session.is_none() {
            self.start_chat_session();
        }

        let relevant_chunks = self.find_relevant_chunks(user_input, 3).await?;
        let context = relevant_chunks
            .iter()
            .map(|chunk| format!("Content from section {}: {}", chunk.section, chunk.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let chat_session = self.chat_session.as_mut().unwrap();

        // Add user message to chat history
        chat_session.add_message("user", user_input);

        // Prepare prompt with context and chat history
        let prompt = format!(
            "You are a helpful assistant analyzing a document. Use the following context to answer the user's question.\n\n\
             Context from the document:\n{}\n\n\
             Chat history:\n{}\n\n\
             Please provide a clear and concise answer based on the context.",
            context,
            chat_session.get_formatted_history()
        );

        // Get response from the model using generateContent
        let response = self.model.send_message(&prompt).await?;
        let answer = response.text();

        // Add assistant's response to chat history
        chat_session.add_message("assistant", &answer);

        Ok(answer)
    }

    pub fn process_pdf(&mut self, path: &str) -> Result<(), ChatError> {
        let text = extract_text(path)?;
        if text.is_empty() {
            return Err(ChatError::NoContent);
        }
        self.process_text(&text)
    }
}

fn calculate_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print_chat_message(role: &str, message: &str) {
        let prefix = match role {
            "user" => "You:".blue().bold(),
            "assistant" => "Assistant:".green().bold(),
            _ => "System:".yellow().bold(),
        };

        println!("\n{}", "─".repeat(100).bright_black());
        println!("{} {}", prefix, message);
        println!("{}", "─".repeat(100).bright_black());
    }

    pub fn print_success(message: &str) {
        println!("\n✓ {}", message.green());
    }

    pub fn print_error(error: &dyn Error) {
        eprintln!("\n{} {}", "Error:".red().bold(), error.to_string().red());
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    println!(
        "{}",
        "🤖 Welcome to PDF Chat Assistant".bright_cyan().bold()
    );
    println!("{}", "═".repeat(50).bright_black());

    // Get PDF path from user
    println!(
        "\n{}",
        "Please enter the path to your PDF file:".bright_yellow()
    );
    let mut pdf_path = String::new();
    std::io::stdin().read_line(&mut pdf_path)?;
    let pdf_path = pdf_path.trim();

    // Check if file exists
    if !std::path::Path::new(pdf_path).exists() {
        println!("{}", "❌ File not found!".bright_red());
        return Ok(());
    }

    println!("\n{}", "🔄 Initializing...".bright_blue());

    let model_generation = "gemini-1.5-flash";
    let model_embedding = "embedding-001";

    // Initialize the document manager
    let model = GenerativeModel::from_env(model_generation)?;
    let mut doc_manager = DocumentChatManager::new(model, model_embedding, 200);
    PrettyPrinter::print_success("Document manager initialized");

    // Process the PDF
    println!("\n{}", "📄 Processing PDF...".bright_blue());
    match doc_manager.process_pdf(pdf_path) {
        Ok(_) => PrettyPrinter::print_success("PDF processed successfully"),
        Err(e) => {
            PrettyPrinter::print_error(&e);
            return Ok(());
        }
    }

    // Generate embeddings
    println!("\n{}", "🧠 Generating embeddings...".bright_blue());
    match doc_manager.generate_embeddings().await {
        Ok(_) => PrettyPrinter::print_success("Embeddings generated successfully"),
        Err(e) => {
            PrettyPrinter::print_error(&e);
            return Ok(());
        }
    }

    // Start chat session
    doc_manager.start_chat_session();
    PrettyPrinter::print_success("Chat session started");

    println!(
        "\n{}",
        "🚀 Ready to chat! Type 'exit' to quit.".bright_green()
    );
    println!("{}", "═".repeat(50).bright_black());

    // Interactive chat loop
    loop {
        // Print prompt
        print!("\n{} ", "You:".blue().bold());
        std::io::stdout().flush()?;

        // Get user input
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Check for exit command
        if input.eq_ignore_ascii_case("exit") {
            println!("\n{}", "👋 Goodbye!".bright_yellow());
            break;
        }

        // Process empty input
        if input.is_empty() {
            println!("{}", "❗ Please enter a question".bright_yellow());
            continue;
        }

        // Show thinking animation
        let thinking_handle = tokio::spawn(async move {
            let thinking_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut i = 0;
            loop {
                print!(
                    "\r{} Thinking... {}",
                    "🤔".bright_blue(),
                    thinking_frames[i].bright_blue()
                );
                std::io::stdout().flush().unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                i = (i + 1) % thinking_frames.len();
            }
        });

        // Get response
        match doc_manager.chat(input).await {
            Ok(response) => {
                // Stop thinking animation
                thinking_handle.abort();
                print!("\r{}", " ".repeat(50)); // Clear thinking animation
                println!("\r");

                // Print response with a typing effect
                print!("{} ", "Assistant:".green().bold());
                for char in response.chars() {
                    print!("{}", char);
                    std::io::stdout().flush()?;
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
                println!("\n");
            }
            Err(e) => {
                thinking_handle.abort();
                PrettyPrinter::print_error(&e);
            }
        }
    }

    Ok(())
}
