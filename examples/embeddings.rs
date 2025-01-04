use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    error::GoogleGenerativeAIError,
    models::{EmbedContentRequest, ModelInfo, TaskType},
};
use serde::{Deserialize, Serialize};
use std::error::Error;
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
}

// Document representation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    title: String,
    text: String,
    embedding: Option<Vec<f32>>,
    metadata: DocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    source: String,
    timestamp: String,
    category: Option<String>,
}

// Document builder for flexible document creation
#[derive(Default)]
pub struct DocumentBuilder {
    title: Option<String>,
    text: Option<String>,
    source: Option<String>,
    category: Option<String>,
}

impl DocumentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn build(self) -> Result<Document, &'static str> {
        let title = self.title.ok_or("Title is required")?;
        let text = self.text.ok_or("Text is required")?;
        let source = self.source.unwrap_or_else(|| "unknown".to_string());

        Ok(Document {
            title,
            text,
            embedding: None,
            metadata: DocumentMetadata {
                source,
                timestamp: chrono::Local::now().to_rfc3339(),
                category: self.category,
            },
        })
    }
}

// Embedding operations
pub struct EmbeddingProcessor {
    model: GenerativeModel,
    model_name: String,
}

impl EmbeddingProcessor {
    pub fn new(model: GenerativeModel, model_name: impl Into<String>) -> Self {
        Self {
            model,
            model_name: model_name.into(),
        }
    }

    pub async fn embed_document(&self, doc: &mut Document) -> Result<(), EmbeddingError> {
        let request = EmbedContentRequest::new(
            &doc.text,
            Some(TaskType::RetrievalDocument),
            Some(doc.title.clone()),
        );
        let response = self.model.embed_content(&self.model_name, request).await?;
        doc.embedding = Some(response.embedding.values);
        Ok(())
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        let request = EmbedContentRequest::new(query, Some(TaskType::RetrievalQuery), None);
        let response = self.model.embed_content(&self.model_name, request).await?;
        Ok(response.embedding.values)
    }

    pub fn find_best_match<'a>(
        &self,
        query_embedding: &[f32],
        documents: &'a [Document],
    ) -> Result<&'a Document, EmbeddingError> {
        if documents.is_empty() {
            return Err(EmbeddingError::NoDocuments);
        }

        documents
            .iter()
            .filter_map(|doc| {
                doc.embedding
                    .as_ref()
                    .map(|emb| (doc, Self::calculate_similarity(query_embedding, emb)))
            })
            .max_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap())
            .map(|(doc, _)| doc)
            .ok_or(EmbeddingError::MissingEmbedding)
    }

    fn calculate_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

// Pretty printing utilities
pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print_document(doc: &Document) {
        println!("\n{}", "─".repeat(80).bright_black());
        println!(
            "{:<15} {}",
            "Title:".blue().bold(),
            doc.title.bright_white()
        );
        println!("{:<15} {}", "Text:".cyan().bold(), doc.text.bright_white());
        println!(
            "{:<15} {}",
            "Source:".yellow().bold(),
            doc.metadata.source.bright_yellow()
        );
        if let Some(category) = &doc.metadata.category {
            println!(
                "{:<15} {}",
                "Category:".magenta().bold(),
                category.bright_magenta()
            );
        }
        println!("{}", "─".repeat(80).bright_black());
    }

    pub fn print_success(message: &str) {
        println!("\n✓ {}", message.green());
    }

    pub fn print_error(error: &dyn Error) {
        eprintln!("\n{} {}", "Error:".red().bold(), error.to_string().red());
    }
}

/// Display detailed information for a model that supports embeddings
fn display_model_info(model: &ModelInfo) {
    println!("\n{}", "─".repeat(100).bright_black());
    println!(
        "{:<20} {}",
        "Model Name:".blue().bold(),
        model.name.bright_blue()
    );
    println!(
        "{:<20} {}",
        "Display Name:".cyan().bold(),
        model.display_name.bright_cyan()
    );
    println!(
        "{:<20} {}",
        "Description:".yellow().bold(),
        model.description.bright_yellow()
    );
    println!(
        "{:<20} {}",
        "Version:".magenta().bold(),
        model.version.bright_magenta()
    );
    println!(
        "{:<20} {}",
        "Methods:".red().bold(),
        model.supported_generation_methods.join(", ").bright_red()
    );
    println!("{}", "─".repeat(100).bright_black());
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();
    PrettyPrinter::print_success("Environment configuration loaded");

    // Initialize the model
    let model = GenerativeModel::from_env("embedding-001")?;
    PrettyPrinter::print_success("Gemini client initialized successfully");

    // List available embedding models
    println!("\n{}", "Gemini Embedding Models".bright_green().bold());
    println!("{}", "═".repeat(50).bright_green());
    println!(
        "{}",
        "Discovering models with embedding capabilities"
            .bright_black()
            .italic()
    );

    match model.list_models().await {
        Ok(models) => {
            let embedding_models: Vec<_> = models
                .models
                .iter()
                .filter(|m| {
                    m.supported_generation_methods
                        .contains(&"embedContent".to_string())
                })
                .collect();

            if embedding_models.is_empty() {
                println!(
                    "{}",
                    "No models supporting embedContent found.".yellow().italic()
                );
            } else {
                for model_info in &embedding_models {
                    display_model_info(model_info);
                }
                println!(
                    "\n{} {}",
                    "Total embedding models found:".bright_blue(),
                    embedding_models.len().to_string().bright_green().bold()
                );
            }
        }
        Err(e) => {
            PrettyPrinter::print_error(&e);
            return Err(e.into());
        }
    }

    // Initialize the embedding processor
    let processor = EmbeddingProcessor::new(model, "embedding-001");
    PrettyPrinter::print_success("Embedding processor initialized");

    // Create sample documents using the builder pattern
    let mut documents = vec![
        DocumentBuilder::new()
            .title("The next generation of AI")
            .text("Gemini API & Google AI Studio: An approachable way to explore and prototype with generative AI applications")
            .source("Google Blog")
            .category("AI Technology")
            .build()?,
        DocumentBuilder::new()
            .title("Self-driving cars")
            .text("Google's self-driving car project, now known as Waymo, uses advanced AI to navigate roads safely.")
            .source("Waymo Documentation")
            .category("Autonomous Vehicles")
            .build()?,
    ];

    // Embed all documents
    println!("\n{}", "Embedding documents...".bright_blue().bold());
    for doc in &mut documents {
        processor.embed_document(doc).await?;
        PrettyPrinter::print_success(&format!("Embedded document: {}", doc.title));
    }

    // Process a query
    let query = "How do you shift gears in the Google car?";
    println!("\n{}", "Processing query...".bright_blue().bold());
    println!("Query: {}", query.bright_white());

    // Find and display the most relevant document
    let query_embedding = processor.embed_query(query).await?;
    match processor.find_best_match(&query_embedding, &documents) {
        Ok(best_match) => {
            println!("\n{}", "Most relevant document:".bright_green().bold());
            PrettyPrinter::print_document(best_match);
        }
        Err(e) => PrettyPrinter::print_error(&e),
    }

    Ok(())
}
