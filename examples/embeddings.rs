use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    error::GoogleGenerativeAIError,
    models::{EmbedContentRequest, TaskType},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, error::Error, fs};
use thiserror::Error;

// Custom error types for better error handling
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Failed to generate embedding: {0}")]
    EmbeddingGeneration(String),
    #[error("No documents available for comparison")]
    NoDocuments,
    #[error("Document has no embedding")]
    MissingEmbedding,
    #[error("API error: {0}")]
    ApiError(#[from] GoogleGenerativeAIError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

// Document types for technical documentation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocType {
    ApiReference,
    Tutorial,
    CodeExample,
    Troubleshooting,
    BestPractices,
}

// Document representation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechDocument {
    title: String,
    content: String,
    embedding: Option<Vec<f32>>,
    metadata: DocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    doc_type: DocType,
    language: String,
    framework: String,
    tags: Vec<String>,
    last_updated: String,
    version: String,
}

// Document builder for flexible document creation
#[derive(Default)]
pub struct TechDocBuilder {
    title: Option<String>,
    content: Option<String>,
    doc_type: Option<DocType>,
    language: Option<String>,
    framework: Option<String>,
    tags: Vec<String>,
    version: Option<String>,
}

impl TechDocBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    pub fn doc_type(mut self, doc_type: DocType) -> Self {
        self.doc_type = Some(doc_type);
        self
    }

    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn framework(mut self, framework: impl Into<String>) -> Self {
        self.framework = Some(framework.into());
        self
    }

    pub fn add_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn build(self) -> Result<TechDocument, &'static str> {
        let title = self.title.ok_or("Title is required")?;
        let content = self.content.ok_or("Content is required")?;
        let doc_type = self.doc_type.ok_or("Document type is required")?;
        let language = self.language.ok_or("Language is required")?;
        let framework = self.framework.unwrap_or_else(|| "None".to_string());
        let version = self.version.unwrap_or_else(|| "latest".to_string());

        Ok(TechDocument {
            title,
            content,
            embedding: None,
            metadata: DocumentMetadata {
                doc_type,
                language,
                framework,
                tags: self.tags,
                last_updated: chrono::Local::now().to_rfc3339(),
                version,
            },
        })
    }
}

// Search result with relevance score
#[derive(Debug)]
pub struct SearchResult<'a> {
    document: &'a TechDocument,
    relevance_score: f32,
}

// Documentation search engine
pub struct DocSearchEngine {
    model: GenerativeModel,
    model_name: String,
    documents: Vec<TechDocument>,
    index: HashMap<DocType, Vec<usize>>, // Index documents by type for faster filtering
}

impl DocSearchEngine {
    pub fn new(model: GenerativeModel, model_name: impl Into<String>) -> Self {
        Self {
            model,
            model_name: model_name.into(),
            documents: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, document: TechDocument) {
        let idx = self.documents.len();
        let doc_type = document.metadata.doc_type.clone();
        self.documents.push(document);
        self.index.entry(doc_type).or_default().push(idx);
    }

    pub async fn embed_documents(&mut self) -> Result<(), EmbeddingError> {
        for doc in &mut self.documents {
            let request = EmbedContentRequest::new(
                &doc.content,
                Some(TaskType::RetrievalDocument),
                Some(doc.title.clone()),
            );
            let response = self.model.embed_content(&self.model_name, request).await?;
            doc.embedding = Some(response.embedding.values);
        }
        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        doc_type_filter: Option<DocType>,
        language_filter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        // Embed the search query
        let request = EmbedContentRequest::new(query, Some(TaskType::RetrievalQuery), None);
        let response = self.model.embed_content(&self.model_name, request).await?;
        let query_embedding = response.embedding.values;

        // Filter and score documents
        let mut results: Vec<SearchResult> = self
            .documents
            .iter()
            .filter(|doc| {
                doc_type_filter
                    .as_ref()
                    .map_or(true, |t| t == &doc.metadata.doc_type)
                    && language_filter
                        .map_or(true, |l| l.eq_ignore_ascii_case(&doc.metadata.language))
            })
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| SearchResult {
                    document: doc,
                    relevance_score: Self::calculate_similarity(&query_embedding, emb),
                })
            })
            .collect();

        // Sort by relevance score
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        Ok(results.into_iter().take(limit).collect())
    }

    fn calculate_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    // Save the document collection to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), EmbeddingError> {
        let json = serde_json::to_string_pretty(&self.documents)?;
        fs::write(path, json)?;
        Ok(())
    }

    // Load the document collection from a file
    pub fn load_from_file(&mut self, path: &str) -> Result<(), EmbeddingError> {
        let json = fs::read_to_string(path)?;
        self.documents = serde_json::from_str(&json)?;

        // Rebuild the index
        self.index.clear();
        for (idx, doc) in self.documents.iter().enumerate() {
            self.index
                .entry(doc.metadata.doc_type.clone())
                .or_default()
                .push(idx);
        }
        Ok(())
    }

    // Function to list and display available embedding models
    pub async fn list_available_models(&self) -> Result<(), Box<dyn Error>> {
        list_embedding_models(&self.model).await
    }
}

// Function to list and display available embedding models
pub async fn list_embedding_models(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "Available Models:".bright_green());
    println!("{}", "═".repeat(50).bright_green());

    let response = model.list_models().await?;

    for model_info in response.models {
        // Only show models that support embeddings
        if model_info
            .supported_generation_methods
            .contains(&"embedContent".to_string())
        {
            println!("\n{}", "─".repeat(80).bright_black());
            println!(
                "{} {}",
                "🤖 Name:".blue().bold(),
                model_info.name.bright_blue()
            );
            println!(
                "{} {}",
                "📋 Display Name:".cyan().bold(),
                model_info.display_name.bright_cyan()
            );
            println!(
                "{} {}",
                "📝 Description:".yellow().bold(),
                model_info.description.bright_yellow()
            );
            println!(
                "{} {}",
                "🔢 Version:".magenta().bold(),
                model_info.version.bright_magenta()
            );

            // Token limits
            println!("\n{}", "📊 Token Limits:".green().bold());
            println!(
                "   {:<20} {}",
                "Input Limit:".white(),
                model_info.input_token_limit.to_string().bright_green()
            );
            println!(
                "   {:<20} {}",
                "Output Limit:".white(),
                model_info.output_token_limit.to_string().bright_green()
            );
        }
    }
    Ok(())
}

// Pretty printing utilities
pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print_document(doc: &TechDocument) {
        println!("\n{}", "─".repeat(100).bright_black());
        println!(
            "{:<15} {}",
            "Title:".blue().bold(),
            doc.title.bright_white()
        );
        println!(
            "{:<15} {}",
            "Type:".cyan().bold(),
            format!("{:?}", doc.metadata.doc_type).bright_cyan()
        );
        println!(
            "{:<15} {}",
            "Language:".yellow().bold(),
            doc.metadata.language.bright_yellow()
        );
        println!(
            "{:<15} {}",
            "Framework:".magenta().bold(),
            doc.metadata.framework.bright_magenta()
        );
        println!(
            "{:<15} {}",
            "Version:".red().bold(),
            doc.metadata.version.bright_red()
        );
        println!(
            "{:<15} {}",
            "Tags:".green().bold(),
            doc.metadata.tags.join(", ").bright_green()
        );
        println!("\n{}", "Content:".blue().bold());
        println!("{}", doc.content.bright_white());
        println!("{}", "─".repeat(100).bright_black());
    }

    pub fn print_search_result(result: &SearchResult) {
        println!("\n{}", "─".repeat(100).bright_black());
        println!(
            "{:<15} {} (Score: {:.4})",
            "Title:".blue().bold(),
            result.document.title.bright_white(),
            result.relevance_score
        );
        println!(
            "{:<15} {}",
            "Type:".cyan().bold(),
            format!("{:?}", result.document.metadata.doc_type).bright_cyan()
        );
        println!(
            "{:<15} {}",
            "Language:".yellow().bold(),
            result.document.metadata.language.bright_yellow()
        );
        // Print first 200 characters of content as preview
        let preview: String = result.document.content.chars().take(200).collect();
        println!("\n{}", "Preview:".blue().bold());
        println!("{}{}", preview.bright_white(), "...".bright_black());
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

    // Initialize the search engine
    let model = GenerativeModel::from_env("embedding-001")?;
    let mut search_engine = DocSearchEngine::new(model, "embedding-001");
    PrettyPrinter::print_success("Search engine initialized");

    // List available embedding models
    println!("\nListing available embedding models...");
    search_engine.list_available_models().await?;

    // Create sample technical documentation
    let documents = vec![
        TechDocBuilder::new()
            .title("Rust Async/Await Tutorial")
            .content("Learn how to write asynchronous code in Rust using async/await. This tutorial covers basic concepts, tokio runtime, and common patterns for handling concurrent operations effectively.")
            .doc_type(DocType::Tutorial)
            .language("Rust")
            .framework("Tokio")
            .add_tag("async")
            .add_tag("concurrency")
            .add_tag("tutorial")
            .version("1.0")
            .build()?,
        TechDocBuilder::new()
            .title("Error Handling Best Practices")
            .content("Comprehensive guide to error handling in Rust. Learn about the Result type, custom error types, error propagation with '?', and how to implement the Error trait.")
            .doc_type(DocType::BestPractices)
            .language("Rust")
            .add_tag("error-handling")
            .add_tag("best-practices")
            .version("2.0")
            .build()?,
        TechDocBuilder::new()
            .title("WebSocket API Reference")
            .content("API documentation for the WebSocket protocol implementation. Includes connection handling, message types, error codes, and security considerations.")
            .doc_type(DocType::ApiReference)
            .language("Rust")
            .framework("Tungstenite")
            .add_tag("websocket")
            .add_tag("api")
            .add_tag("networking")
            .version("3.1")
            .build()?,
        TechDocBuilder::new()
            .title("Common Concurrency Bugs")
            .content("Troubleshooting guide for common concurrency issues in Rust. Covers deadlocks, race conditions, and how to debug them using tools like MIRI and thread sanitizer.")
            .doc_type(DocType::Troubleshooting)
            .language("Rust")
            .add_tag("debugging")
            .add_tag("concurrency")
            .version("1.2")
            .build()?,
        TechDocBuilder::new()
            .title("WebSocket Chat Example")
            .content(r#"
                ```rust
                use tokio::net::{TcpListener, TcpStream};
                use futures::{StreamExt, SinkExt};
                
                async fn handle_connection(stream: TcpStream) {
                    let ws_stream = tokio_tungstenite::accept_async(stream)
                        .await
                        .expect("Failed to accept");
                    
                    let (write, read) = ws_stream.split();
                    read.forward(write).await.expect("Failed to forward");
                }
                ```
                Complete example of building a real-time chat application using WebSockets in Rust."#)
            .doc_type(DocType::CodeExample)
            .language("Rust")
            .framework("Tokio")
            .add_tag("websocket")
            .add_tag("example")
            .add_tag("chat")
            .version("1.0")
            .build()?,
    ];

    // Add documents to the search engine
    for doc in documents {
        search_engine.add_document(doc);
    }

    // Embed all documents
    println!("\n{}", "Embedding documents...".bright_blue());
    search_engine.embed_documents().await?;
    PrettyPrinter::print_success("Documents embedded successfully");

    // Perform sample searches
    println!("\n{}", "Search Examples:".bright_green());
    println!("{}", "═".repeat(50).bright_green());

    // Example 1: Search for concurrency-related documentation
    let query = "How to handle concurrent operations in Rust?";
    println!("\nQuery: {}", query.bright_white());
    let results = search_engine.search(query, None, Some("Rust"), 3).await?;

    println!("\nTop {} results for concurrency:", results.len());
    for result in &results {
        PrettyPrinter::print_search_result(result);
    }

    // Example 2: Search for WebSocket examples
    let query = "Show me a WebSocket implementation example";
    println!("\nQuery: {}", query.bright_white());
    let results = search_engine
        .search(query, Some(DocType::CodeExample), None, 2)
        .await?;

    println!("\nWebSocket examples:");
    for result in &results {
        PrettyPrinter::print_search_result(result);
    }

    // Example 3: Search for error handling documentation
    let query = "Best practices for handling errors";
    println!("\nQuery: {}", query.bright_white());
    let results = search_engine
        .search(query, Some(DocType::BestPractices), Some("Rust"), 2)
        .await?;

    println!("\nError handling guides:");
    for result in &results {
        PrettyPrinter::print_search_result(result);
    }

    // Save the document collection
    search_engine.save_to_file("tech_docs.json")?;
    PrettyPrinter::print_success("Document collection saved to tech_docs.json");

    Ok(())
}
