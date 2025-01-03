use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request},
    GenerativeModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    error::Error,
    time::{Duration, Instant},
};

/// A simple in-memory cache for demonstration
#[derive(Default)]
struct ResponseCache {
    cache: HashMap<String, String>,
}

impl ResponseCache {
    fn new() -> Self {
        Self::default()
    }

    fn get(&self, key: &str) -> Option<&String> {
        self.cache.get(key)
    }

    fn set(&mut self, key: String, value: String) {
        self.cache.insert(key, value);
    }

    fn stats(&self) -> (usize, usize) {
        let total_size = self.cache.values().map(|v| v.len()).sum();
        (self.cache.len(), total_size)
    }
}

/// Demonstrate caching with different queries
async fn demonstrate_caching(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    let mut cache = ResponseCache::new();
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈")
            .template("{spinner:.green} {msg}")?,
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let test_queries = [
        ("Basic Math", "What is 2 + 2?", "Simple arithmetic question"),
        (
            "Science",
            "What is photosynthesis?",
            "Basic science concept",
        ),
        (
            "Duplicate Math",
            "What is 2 + 2?",
            "Repeated query to demonstrate caching",
        ),
        (
            "History",
            "Who was Albert Einstein?",
            "Historical figure query",
        ),
        (
            "Duplicate Science",
            "What is photosynthesis?",
            "Another repeated query",
        ),
    ];

    for (query_name, query_text, description) in test_queries {
        println!("\n{}", "━".repeat(100).bright_black());
        println!("{} {}", "🔍 Query:".blue().bold(), query_name.bright_blue());
        println!(
            "{} {}",
            "📋 Description:".cyan().bold(),
            description.bright_cyan()
        );
        println!("{}", "─".repeat(100).bright_black());

        println!(
            "{} {}",
            "❓ Question:".yellow().bold(),
            query_text.bright_yellow()
        );

        // Check cache first
        let start_time = Instant::now();
        if let Some(cached_response) = cache.get(query_text) {
            let elapsed = start_time.elapsed();
            println!(
                "\n{} {}",
                "📦 Cache Hit!".green().bold(),
                format!("Retrieved in {:.2?}", elapsed).bright_green()
            );
            println!("{}", cached_response.white());
        } else {
            pb.set_message("Generating response...");
            let request = Request::builder()
                .contents(vec![Content {
                    role: None,
                    parts: vec![Part::Text {
                        text: query_text.into(),
                    }],
                }])
                .build();

            match model.generate_response(request).await {
                Ok(response) => {
                    let elapsed = start_time.elapsed();
                    pb.finish_and_clear();
                    println!(
                        "\n{} {}",
                        "🌟 New Response:".magenta().bold(),
                        format!("Generated in {:.2?}", elapsed).bright_magenta()
                    );
                    let response_text = response.text();
                    println!("{}", response_text.white());

                    // Cache the response
                    cache.set(query_text.to_string(), response_text);

                    // Display cache stats
                    let (entries, size) = cache.stats();
                    println!(
                        "\n{} {} entries, {} bytes",
                        "📊 Cache Stats:".bright_white().bold(),
                        entries,
                        size
                    );
                }
                Err(e) => {
                    pb.finish_and_clear();
                    println!(
                        "{} {}",
                        "❌ Error:".red().bold(),
                        format!("Failed to generate response: {}", e).red()
                    );
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "📦 Gemini Response Cache Demo".bright_green().bold());
    println!("{}", "=========================".bright_green());
    println!(
        "{}",
        "Demonstrating response caching with performance metrics"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Initialize the model
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run caching demonstrations
    demonstrate_caching(&model).await?;

    println!(
        "\n{}",
        "✨ Cache demonstration completed!".bright_green().bold()
    );
    Ok(())
}
