use dotenv::dotenv;
use gemini_ai_rust::{
    file::{FileState, GoogleAIFileManager},
    models::{Content, Part, Request, Role},
    GenerativeModel,
};
use std::{error::Error, io::Write, path::Path};
use tokio_stream::StreamExt;

// Get absolute path to example image
const FILE_PATH: &str = "examples/test.txt";

async fn file_from_path(file_path: impl AsRef<Path>) -> std::io::Result<Part> {
    // Create a new file manager instance
    let file_manager = GoogleAIFileManager::from_env();

    // Check if file exists
    let file_path = file_path.as_ref();
    if !file_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found",
        ));
    }

    // Upload file and get its info
    let file_info = file_manager
        .upload_file(file_path, "Example Text File".to_string())
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // If file is still processing, wait for it to complete
    if matches!(file_info.state, FileState::Processing) {
        let processed_file = file_manager
            .wait_for_file_processing(&file_info.name, 10, 1000)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Part::file_data(
            processed_file.mime_type,
            processed_file.uri,
        ))
    } else {
        Ok(Part::file_data(file_info.mime_type, file_info.uri))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-2.0-flash-exp")?;

    // Create a request with some text
    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![
                Part::text("Please describe this file."),
                file_from_path(FILE_PATH).await?,
            ],
        }])
        .build();

    // Stream the response
    let mut stream = model.stream_generate_response(request).await?;

    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => {
                print!("{}", response.text());
                std::io::stdout().flush().unwrap();
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    Ok(())
}
