use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{client::GenerativeModel, models::EmbedContentRequest};
use std::error::Error;

/// Display detailed information for a model that supports embeddings
fn display_model_info(model: &gemini_ai_rust::models::ModelInfo) {
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

/// Display embedding results in a formatted way
fn display_embedding_result(prompt: &str, embedding: &[f32]) {
    println!("\n{}", "━".repeat(100).bright_black());
    println!(
        "{:<15} {}",
        "Input Text:".blue().bold(),
        prompt.bright_white()
    );

    println!(
        "{:<15} {}",
        "Dimensions:".yellow().bold(),
        embedding.len().to_string().bright_yellow()
    );

    // Display first few and last few dimensions
    let preview_size = 3;
    println!("{}", "Embedding Vector:".magenta().bold());

    // First few dimensions
    for (i, value) in embedding.iter().take(preview_size).enumerate() {
        println!(
            "  {:<4} │ {:.6}",
            format!("[{}]", i).bright_black(),
            value.to_string().bright_cyan()
        );
    }

    // Middle ellipsis if there are more dimensions
    if embedding.len() > preview_size * 2 {
        println!("  {}", "   ⋮".bright_black());
    }

    // Last few dimensions
    for (i, value) in embedding.iter().rev().take(preview_size).rev().enumerate() {
        let idx = embedding.len() - preview_size + i;
        println!(
            "  {:<4} │ {:.6}",
            format!("[{}]", idx).bright_black(),
            value.to_string().bright_cyan()
        );
    }

    println!("{}", "━".repeat(100).bright_black());
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("\n{}", "Gemini Embedding Models".bright_green().bold());
    println!("{}", "═".repeat(50).bright_green());
    println!(
        "{}",
        "Discovering models with embedding capabilities"
            .bright_black()
            .italic()
    );

    dotenv().ok();
    println!("{}", "✓ Environment configuration loaded".green());

    let model = GenerativeModel::from_env("text-embedding-004")?;
    println!("{}", "✓ Gemini client initialized successfully".green());

    println!("\n{}", "Available Embedding Models:".bright_blue().bold());
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
            eprintln!(
                "{} {}",
                "Error:".red().bold(),
                format!("Failed to retrieve models: {}", e).red()
            );
            return Err(e.into());
        }
    }

    let model_name = "text-embedding-004";
    println!(
        "\n{} {}",
        "Generating embeddings for:".bright_blue().bold(),
        "Hello world".bright_white()
    );

    let response = model
        .embed_content(
            model_name,
            EmbedContentRequest::new("Hello world", None, None),
        )
        .await?;

    display_embedding_result("Hello world", &response.embedding.values);

    println!(
        "\n{} {}",
        "✓".green(),
        "Embedding generation completed successfully".green()
    );

    Ok(())
}
