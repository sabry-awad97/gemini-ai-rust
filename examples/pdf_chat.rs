use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    error::GoogleGenerativeAIError,
    models::{EmbedContentRequest, TaskType},
};
use serde::{Deserialize, Serialize};
use std::error::Error;
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
    PrettyPrinter::print_success("Environment configuration loaded");

    // Sample document content
    let document_content = r#"
        This is a sample document about Rust programming.

        Rust is a systems programming language that runs blazingly fast, prevents segfaults,
        and guarantees thread safety. Rust's rich type system and ownership model guarantee
        memory-safety and thread-safety — enabling you to eliminate many classes of bugs at
        compile-time.

        Key Features of Rust:
        1. Zero-cost abstractions
        2. Move semantics
        3. Guaranteed memory safety
        4. Threads without data races
        5. Trait-based generics
        6. Pattern matching
        7. Type inference
        8. Minimal runtime
        9. Efficient C bindings

        Rust is used in various domains:
        - Systems programming
        - WebAssembly
        - Command line tools
        - Network services
        - Embedded systems
        
        The Rust compiler plays a gatekeeper role in ensuring memory safety.
        It enforces strict rules about borrowing and ownership, which might seem
        restrictive at first but help prevent common programming errors.

        Memory Management in Rust:
        Rust uses a unique ownership system to manage memory. Each value has an owner,
        and there can only be one owner at a time. When the owner goes out of scope,
        the value is dropped. This system ensures memory safety without garbage collection.

        Error Handling:
        Rust encourages explicit error handling through the Result type. This makes
        error cases visible and ensures they are handled appropriately. The ? operator
        makes working with Results ergonomic.

        Concurrency:
        Rust's ownership and type systems enable fearless concurrency. By enforcing strict
        rules at compile time, Rust prevents data races and other concurrent programming pitfalls.
    "#;

    let model_generation = "gemini-1.5-flash";
    let model_embedding = "embedding-001";

    // Initialize the document manager
    let model = GenerativeModel::from_env(model_generation)?;
    let mut doc_manager = DocumentChatManager::new(model, model_embedding, 200);
    PrettyPrinter::print_success("Document manager initialized");

    // Process the document
    doc_manager.process_text(document_content)?;
    PrettyPrinter::print_success("Document processed successfully");

    // Generate embeddings
    println!("\nGenerating embeddings...");
    doc_manager.generate_embeddings().await?;
    PrettyPrinter::print_success("Embeddings generated successfully");

    // Start chat session
    doc_manager.start_chat_session();
    PrettyPrinter::print_success("Chat session started");

    // Example chat interactions
    let questions = vec![
        "What are the key features of Rust?",
        "How does Rust handle memory management?",
        "Can you explain Rust's approach to concurrency?",
        "What domains is Rust commonly used in?",
    ];

    println!("\n{}", "Chat Examples:".bright_green());
    println!("{}", "═".repeat(50).bright_green());

    for question in questions {
        PrettyPrinter::print_chat_message("user", question);
        let response = doc_manager.chat(question).await?;
        PrettyPrinter::print_chat_message("assistant", &response);
    }

    Ok(())
}
