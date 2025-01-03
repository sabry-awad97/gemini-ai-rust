use base64::{engine::general_purpose::STANDARD as base64_engine, Engine};
use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, InlineData, Part, Request},
    GenerativeModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{error::Error, path::Path, time::Duration};

const IMAGE_PATH: &str = "examples/file_data.png";
const TEST_FILE: &str = "examples/test.txt";

/// Create a sample text file for demonstration
fn create_test_file() -> Result<(), Box<dyn Error>> {
    println!("{}", "📝 Creating test file...".yellow().bold());
    std::fs::write(
        TEST_FILE,
        r#"This is a test file for the Gemini AI API file upload functionality.
It contains some sample text that we can use to verify the file upload, download, and processing features."#,
    )?;
    println!("{} {}", "✓".green(), "Test file created".green());
    Ok(())
}

/// Clean up test files after demonstration
fn cleanup_test_file() -> Result<(), Box<dyn Error>> {
    if Path::new(TEST_FILE).exists() {
        std::fs::remove_file(TEST_FILE)?;
        println!("{} {}", "✓".green(), "Test file cleaned up".green());
    }
    Ok(())
}

/// Demonstrate file operations with progress indicators
async fn demonstrate_file_operations(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    // Setup progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈")
            .template("{spinner:.green} {msg}")?,
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    // 1. Text File Analysis
    println!("\n{}", "📄 Text File Analysis".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    pb.set_message("Reading text file...");
    let text_content = std::fs::read_to_string(TEST_FILE)?;
    pb.finish_and_clear();

    println!(
        "{} {}",
        "📊 File Stats:".yellow().bold(),
        format!(
            "{} lines, {} characters",
            text_content.lines().count(),
            text_content.chars().count()
        )
        .bright_yellow()
    );

    let text_request = Request::builder()
        .contents(vec![Content {
            role: None,
            parts: vec![Part::Text {
                text: format!(
                    "Analyze this text content and provide insights: {}",
                    text_content
                ),
            }],
        }])
        .build();

    pb.set_message("Analyzing text content...");
    match model.generate_response(text_request).await {
        Ok(response) => {
            pb.finish_and_clear();
            println!("\n{}", "📝 Text Analysis:".green().bold());
            println!("{}", response.text().white());
        }
        Err(e) => {
            pb.finish_and_clear();
            println!(
                "{} {}",
                "❌ Error:".red().bold(),
                format!("Failed to analyze text: {}", e).red()
            );
        }
    }

    // 2. Image Analysis
    println!("\n{}", "🖼️  Image Analysis".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    if !Path::new(IMAGE_PATH).exists() {
        println!(
            "{} {}",
            "⚠️ Note:".yellow().bold(),
            format!("Image file not found at: {}", IMAGE_PATH).yellow()
        );
        return Ok(());
    }

    pb.set_message("Reading image file...");
    let image_data = std::fs::read(IMAGE_PATH)?;
    let mime_type = mime_guess::from_path(IMAGE_PATH)
        .first_or_octet_stream()
        .to_string();
    pb.finish_and_clear();

    println!(
        "{} {}",
        "📊 Image Stats:".yellow().bold(),
        format!(
            "{:.2} MB, MIME type: {}",
            image_data.len() as f64 / 1_048_576.0,
            mime_type
        )
        .bright_yellow()
    );

    let image_request = Request::builder()
        .contents(vec![Content {
            role: None,
            parts: vec![
                Part::Text {
                    text: "Describe this image in detail:".into(),
                },
                Part::InlineData {
                    inline_data: InlineData {
                        mime_type,
                        data: base64_engine.encode(image_data),
                    },
                },
            ],
        }])
        .build();

    pb.set_message("Analyzing image content...");
    match model.generate_response(image_request).await {
        Ok(response) => {
            pb.finish_and_clear();
            println!("\n{}", "🎨 Image Analysis:".green().bold());
            println!("{}", response.text().white());
        }
        Err(e) => {
            pb.finish_and_clear();
            println!(
                "{} {}",
                "❌ Error:".red().bold(),
                format!("Failed to analyze image: {}", e).red()
            );
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🗂️  Gemini File Analysis Demo".bright_green().bold());
    println!("{}", "=========================".bright_green());
    println!(
        "{}",
        "Demonstrating text and image file analysis capabilities"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create test file
    create_test_file()?;

    // Initialize the model
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run file operation demonstrations
    demonstrate_file_operations(&model).await?;

    // Cleanup
    cleanup_test_file()?;

    println!("\n{}", "✨ File analysis completed!".bright_green().bold());
    Ok(())
}
