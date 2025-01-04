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

// Custom error types for pharmacy management system
#[derive(Error, Debug)]
pub enum PharmacyError {
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

// Document types for pharmacy documentation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PharmacyDocType {
    DrugInfo,
    Procedure,
    Policy,
    Regulation,
    SideEffect,
    Interaction,
    Storage,
}

// Document representation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmacyDocument {
    title: String,
    content: String,
    embedding: Option<Vec<f32>>,
    metadata: PharmacyMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmacyMetadata {
    doc_type: PharmacyDocType,
    category: String,
    drug_class: Option<String>,
    requires_prescription: bool,
    last_updated: String,
    version: String,
    tags: Vec<String>,
}

// Document builder for pharmacy documents
#[derive(Default)]
pub struct PharmacyDocBuilder {
    title: Option<String>,
    content: Option<String>,
    doc_type: Option<PharmacyDocType>,
    category: Option<String>,
    drug_class: Option<String>,
    requires_prescription: Option<bool>,
    tags: Vec<String>,
    version: Option<String>,
}

impl PharmacyDocBuilder {
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

    pub fn doc_type(mut self, doc_type: PharmacyDocType) -> Self {
        self.doc_type = Some(doc_type);
        self
    }

    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn drug_class(mut self, drug_class: impl Into<String>) -> Self {
        self.drug_class = Some(drug_class.into());
        self
    }

    pub fn requires_prescription(mut self, requires_prescription: bool) -> Self {
        self.requires_prescription = Some(requires_prescription);
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

    pub fn build(self) -> Result<PharmacyDocument, &'static str> {
        let title = self.title.ok_or("Title is required")?;
        let content = self.content.ok_or("Content is required")?;
        let doc_type = self.doc_type.ok_or("Document type is required")?;
        let category = self.category.ok_or("Category is required")?;
        let requires_prescription = self.requires_prescription.unwrap_or(false);
        let version = self.version.unwrap_or_else(|| "1.0".to_string());

        Ok(PharmacyDocument {
            title,
            content,
            embedding: None,
            metadata: PharmacyMetadata {
                doc_type,
                category,
                drug_class: self.drug_class,
                requires_prescription,
                last_updated: chrono::Local::now().to_rfc3339(),
                version,
                tags: self.tags,
            },
        })
    }
}

// Search result with relevance score
#[derive(Debug)]
pub struct SearchResult<'a> {
    document: &'a PharmacyDocument,
    relevance_score: f32,
}

// Pharmacy documentation search engine
pub struct PharmacySearchEngine {
    model: GenerativeModel,
    model_name: String,
    documents: Vec<PharmacyDocument>,
    index: HashMap<PharmacyDocType, Vec<usize>>,
}

impl PharmacySearchEngine {
    pub fn new(model: GenerativeModel, model_name: impl Into<String>) -> Self {
        Self {
            model,
            model_name: model_name.into(),
            documents: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, document: PharmacyDocument) {
        let idx = self.documents.len();
        let doc_type = document.metadata.doc_type.clone();
        self.documents.push(document);
        self.index.entry(doc_type).or_default().push(idx);
    }

    pub async fn embed_documents(&mut self) -> Result<(), PharmacyError> {
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
        doc_type_filter: Option<PharmacyDocType>,
        category_filter: Option<&str>,
        prescription_only: Option<bool>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, PharmacyError> {
        let request = EmbedContentRequest::new(query, Some(TaskType::RetrievalQuery), None);
        let response = self.model.embed_content(&self.model_name, request).await?;
        let query_embedding = response.embedding.values;

        let mut results: Vec<SearchResult> = self
            .documents
            .iter()
            .filter(|doc| {
                doc_type_filter
                    .as_ref()
                    .map_or(true, |t| t == &doc.metadata.doc_type)
                    && category_filter
                        .map_or(true, |c| c.eq_ignore_ascii_case(&doc.metadata.category))
                    && prescription_only.map_or(true, |p| p == doc.metadata.requires_prescription)
            })
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| SearchResult {
                    document: doc,
                    relevance_score: Self::calculate_similarity(&query_embedding, emb),
                })
            })
            .collect();

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        Ok(results.into_iter().take(limit).collect())
    }

    fn calculate_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), PharmacyError> {
        let json = serde_json::to_string_pretty(&self.documents)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(&mut self, path: &str) -> Result<(), PharmacyError> {
        let json = fs::read_to_string(path)?;
        self.documents = serde_json::from_str(&json)?;

        self.index.clear();
        for (idx, doc) in self.documents.iter().enumerate() {
            self.index
                .entry(doc.metadata.doc_type.clone())
                .or_default()
                .push(idx);
        }
        Ok(())
    }
}

// Pretty printing utilities
pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print_document(doc: &PharmacyDocument) {
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
            "Category:".yellow().bold(),
            doc.metadata.category.bright_yellow()
        );
        if let Some(drug_class) = &doc.metadata.drug_class {
            println!(
                "{:<15} {}",
                "Drug Class:".magenta().bold(),
                drug_class.bright_magenta()
            );
        }
        println!(
            "{:<15} {}",
            "Prescription:".red().bold(),
            doc.metadata.requires_prescription.to_string().bright_red()
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
            "Category:".yellow().bold(),
            result.document.metadata.category.bright_yellow()
        );
        if let Some(drug_class) = &result.document.metadata.drug_class {
            println!(
                "{:<15} {}",
                "Drug Class:".magenta().bold(),
                drug_class.bright_magenta()
            );
        }
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
    let mut search_engine = PharmacySearchEngine::new(model, "embedding-001");
    PrettyPrinter::print_success("Search engine initialized");

    // Create sample pharmacy documentation
    let documents = vec![
        PharmacyDocBuilder::new()
            .title("Amoxicillin Usage Guidelines")
            .content("Amoxicillin is a penicillin antibiotic used to treat various bacterial infections. \
                     Recommended dosage: 250-500mg every 8 hours for adults. Take with or without food. \
                     Complete the full course even if symptoms improve. Store at room temperature away from moisture.")
            .doc_type(PharmacyDocType::DrugInfo)
            .category("Antibiotics")
            .drug_class("Penicillin")
            .requires_prescription(true)
            .add_tag("antibiotic")
            .add_tag("bacterial-infection")
            .version("2.1")
            .build()?,

        PharmacyDocBuilder::new()
            .title("Prescription Verification Procedure")
            .content("Standard procedure for verifying prescriptions: 1. Check patient identification \
                     2. Verify prescriber's credentials and DEA number 3. Validate prescription format and content \
                     4. Check for drug interactions 5. Document verification process 6. Contact prescriber if needed")
            .doc_type(PharmacyDocType::Procedure)
            .category("Operations")
            .requires_prescription(false)
            .add_tag("verification")
            .add_tag("safety")
            .add_tag("compliance")
            .version("3.0")
            .build()?,

        PharmacyDocBuilder::new()
            .title("Drug Storage Requirements")
            .content("Guidelines for proper medication storage: Maintain temperature between 20-25°C (68-77°F). \
                     Protect from light and moisture. Refrigerate items marked accordingly. \
                     Monitor temperature logs daily. Separate controlled substances in secured storage.")
            .doc_type(PharmacyDocType::Storage)
            .category("Inventory")
            .requires_prescription(false)
            .add_tag("storage")
            .add_tag("temperature")
            .add_tag("safety")
            .version("1.5")
            .build()?,

        PharmacyDocBuilder::new()
            .title("Ibuprofen Drug Interactions")
            .content("Important drug interactions with Ibuprofen: 1. Aspirin - may decrease cardiovascular benefits \
                     2. Blood pressure medications - may decrease effectiveness 3. Anticoagulants - increased bleeding risk \
                     4. SSRIs - increased risk of bleeding 5. Lithium - increased lithium levels")
            .doc_type(PharmacyDocType::Interaction)
            .category("NSAIDs")
            .drug_class("NSAID")
            .requires_prescription(false)
            .add_tag("interaction")
            .add_tag("safety")
            .add_tag("NSAID")
            .version("2.0")
            .build()?,

        PharmacyDocBuilder::new()
            .title("Controlled Substances Policy")
            .content("Policy for handling controlled substances: Must be stored in double-locked cabinet. \
                     Maintain perpetual inventory. Perform daily counts of Schedule II substances. \
                     Document all discrepancies and report to supervisor immediately. \
                     Verify prescriber DEA number for all controlled substance prescriptions.")
            .doc_type(PharmacyDocType::Policy)
            .category("Compliance")
            .requires_prescription(true)
            .add_tag("controlled")
            .add_tag("policy")
            .add_tag("security")
            .version("4.0")
            .build()?,
    ];

    // Add documents to the search engine
    for doc in documents {
        search_engine.add_document(doc);
    }

    // Embed all documents
    println!("\n{}", "Embedding pharmacy documents...".bright_blue());
    search_engine.embed_documents().await?;
    PrettyPrinter::print_success("Documents embedded successfully");

    // Perform sample searches
    println!("\n{}", "Search Examples:".bright_green());
    println!("{}", "═".repeat(50).bright_green());

    // Example 1: Search for antibiotic information
    let query = "What is the recommended dosage for antibiotics?";
    println!("\nQuery: {}", query.bright_white());
    let results = search_engine
        .search(
            query,
            Some(PharmacyDocType::DrugInfo),
            Some("Antibiotics"),
            None,
            2,
        )
        .await?;

    println!("\nAntibiotic Information:");
    for result in &results {
        PrettyPrinter::print_search_result(result);
    }

    // Example 2: Search for storage requirements
    let query = "How should medications be stored properly?";
    println!("\nQuery: {}", query.bright_white());
    let results = search_engine
        .search(query, Some(PharmacyDocType::Storage), None, None, 2)
        .await?;

    println!("\nStorage Guidelines:");
    for result in &results {
        PrettyPrinter::print_search_result(result);
    }

    // Example 3: Search for prescription medication policies
    let query = "What are the requirements for handling controlled substances?";
    println!("\nQuery: {}", query.bright_white());
    let results = search_engine
        .search(query, Some(PharmacyDocType::Policy), None, Some(true), 2)
        .await?;

    println!("\nPrescription Policies:");
    for result in &results {
        PrettyPrinter::print_search_result(result);
    }

    // Save the document collection
    search_engine.save_to_file("target/pharmacy_docs.json")?;
    PrettyPrinter::print_success("Pharmacy documentation saved to target/pharmacy_docs.json");

    Ok(())
}
