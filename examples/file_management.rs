use dotenv::dotenv;
use gemini_ai_rust::file::{FileState, GoogleAIFileManager};
use std::path::PathBuf;

async fn delete_all_files(
    file_manager: &GoogleAIFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nListing all files to delete:");
    let files = file_manager.list_files().await?;
    for file in files {
        println!(
            "Deleting: {} ({})",
            file.display_name.as_deref().unwrap_or(&file.name),
            file.state
        );
        file_manager.delete_file(&file.name).await?;
    }
    println!("All files deleted successfully!");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Get API key from environment
    let api_key =
        std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable must be set");

    // Create a new file manager instance
    let file_manager = GoogleAIFileManager::new(api_key);

    // Delete all existing files first
    delete_all_files(&file_manager).await?;

    // Example file path
    let example_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("test.txt");

    println!("Uploading file: {:?}", example_file);

    // Upload the file with display name
    let file_info = file_manager
        .upload_file(&example_file, Some("Example Text File".to_string()))
        .await?;
    println!("File uploaded successfully!");
    println!("File Info:");
    println!("  Name: {}", file_info.name);
    println!("  URI: {}", file_info.uri);
    println!("  Size: {} bytes", file_info.size_bytes);
    println!("  MIME Type: {}", file_info.mime_type);
    println!("  State: {}", file_info.state);

    // Wait for processing if needed
    if matches!(file_info.state, FileState::Processing) {
        println!("Waiting for file processing...");
        let processed_file = file_manager
            .wait_for_file_processing(&file_info.name, 10, 1000)
            .await?;
        println!("File processing complete! State: {}", processed_file.state);
    }

    // List all files
    println!("\nListing all files:");
    let files = file_manager.list_files().await?;
    for file in &files {
        println!(
            "- {} (internal name: {}) (state: {})",
            file.display_name.as_deref().unwrap_or(&file.name),
            file.name,
            file.state
        );
    }

    // Clean up by deleting the uploaded file
    println!("\nDeleting file: {}", file_info.name);
    file_manager.delete_file(&file_info.name).await?;
    println!("File deleted successfully!");

    Ok(())
}
