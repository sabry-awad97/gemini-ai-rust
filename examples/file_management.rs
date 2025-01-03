use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    file::{FileState, GoogleAIFileManager},
    models::{Content, Part, Request},
    GenerativeModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{error::Error, path::Path, time::Duration};

const TEST_FILE: &str = "examples/test_file.txt";
const TEST_IMAGE: &str = "examples/test_image.jpg";

/// Create test files for demonstration
fn create_test_files() -> Result<(), Box<dyn Error>> {
    println!("{}", "📝 Creating test files...".yellow().bold());

    // Create text file
    std::fs::write(
        TEST_FILE,
        "This is a test file for the Gemini AI File Management demo.\n\
         It demonstrates uploading, processing, and analyzing files.\n\
         The file contains multiple lines of text for better demonstration.",
    )?;
    println!("{} {}", "✓".green(), "Text file created".green());

    // Create a simple test image (a 1x1 pixel black JPEG)
    let image_data = [
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00,
        0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00,
        0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08,
        0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7F, 0x00, 0xFF, 0xD9,
    ];
    std::fs::write(TEST_IMAGE, image_data)?;
    println!("{} {}", "✓".green(), "Test image created".green());

    Ok(())
}

/// Clean up test files after demonstration
fn cleanup_test_files() -> Result<(), Box<dyn Error>> {
    for file in [TEST_FILE, TEST_IMAGE] {
        if Path::new(file).exists() {
            std::fs::remove_file(file)?;
            println!("{} {}", "✓".green(), format!("Cleaned up {}", file).green());
        }
    }
    Ok(())
}

/// Demonstrate file management operations
async fn demonstrate_file_management(
    model: &GenerativeModel,
    file_manager: &GoogleAIFileManager,
) -> Result<(), Box<dyn Error>> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈")
            .template("{spinner:.green} {msg}")?,
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    // 1. File Upload and Processing
    println!("\n{}", "📤 File Upload Demo".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    // Upload text file
    pb.set_message("Uploading text file...");
    let text_file_info = file_manager
        .upload_file(TEST_FILE, "Example Text File".to_string())
        .await?;
    pb.finish_and_clear();

    println!(
        "{} {}",
        "📄 Text File Info:".yellow().bold(),
        text_file_info.name.bright_yellow()
    );
    println!(
        "   {:<15} {}",
        "MIME Type:".white(),
        text_file_info.mime_type.bright_white()
    );
    println!(
        "   {:<15} {}",
        "State:".white(),
        format!("{:?}", text_file_info.state).bright_white()
    );

    // Wait for processing if needed
    if matches!(text_file_info.state, FileState::Processing) {
        pb.set_message("Waiting for text file processing...");
        let processed_file = file_manager
            .wait_for_file_processing(&text_file_info.name, 10, 1000)
            .await?;
        pb.finish_and_clear();
        println!(
            "{} {}",
            "✓ Processing complete:".green().bold(),
            processed_file.uri.bright_green()
        );
    }

    // Upload image file
    println!("\n{}", "🖼️  Image Upload Demo".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    pb.set_message("Uploading image file...");
    let image_file_info = file_manager
        .upload_file(TEST_IMAGE, "Example Image File".to_string())
        .await?;
    pb.finish_and_clear();

    println!(
        "{} {}",
        "🎨 Image File Info:".yellow().bold(),
        image_file_info.name.bright_yellow()
    );
    println!(
        "   {:<15} {}",
        "MIME Type:".white(),
        image_file_info.mime_type.bright_white()
    );
    println!(
        "   {:<15} {}",
        "State:".white(),
        format!("{:?}", image_file_info.state).bright_white()
    );

    // 2. File Listing
    println!("\n{}", "📋 File Listing Demo".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    pb.set_message("Listing all files...");
    let files = file_manager.list_files().await?;
    pb.finish_and_clear();

    println!("{}", "📚 Available Files:".magenta().bold());
    for file in &files {
        println!(
            "   {} {}",
            "•".bright_magenta(),
            format!("{} ({:?}, {})", file.name, file.state, file.mime_type).bright_white()
        );
    }

    // 3. File Analysis
    println!("\n{}", "🔍 File Analysis Demo".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    let request = Request::builder()
        .contents(vec![Content {
            role: None,
            parts: vec![
                Part::Text {
                    text: "Analyze this file:".into(),
                },
                Part::file_data(text_file_info.mime_type, text_file_info.uri),
            ],
        }])
        .build();

    pb.set_message("Analyzing file content...");
    match model.generate_response(request).await {
        Ok(response) => {
            pb.finish_and_clear();
            println!("\n{}", "📝 Analysis Results:".green().bold());
            println!("{}", response.text().white());
        }
        Err(e) => {
            pb.finish_and_clear();
            println!(
                "{} {}",
                "❌ Error:".red().bold(),
                format!("Failed to analyze file: {}", e).red()
            );
        }
    }

    // 4. File Deletion
    println!("\n{}", "🗑️  File Cleanup Demo".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    for file in &files {
        pb.set_message(format!("Deleting {}...", file.name));
        file_manager.delete_file(&file.name).await?;
        pb.finish_and_clear();
        println!(
            "{} {}",
            "✓".green(),
            format!("Deleted {}", file.name).green()
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "{}",
        "🗂️  Gemini File Management Demo".bright_green().bold()
    );
    println!("{}", "==========================".bright_green());
    println!(
        "{}",
        "Demonstrating file upload, processing, and analysis capabilities"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create test files
    create_test_files()?;

    // Initialize managers
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    let file_manager = GoogleAIFileManager::from_env();
    println!("{}", "✓ Gemini managers initialized".green());

    // Run file management demonstrations
    demonstrate_file_management(&model, &file_manager).await?;

    // Cleanup test files
    cleanup_test_files()?;

    println!(
        "\n{}",
        "✨ File management demo completed!".bright_green().bold()
    );
    Ok(())
}
