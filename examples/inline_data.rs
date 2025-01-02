use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Role},
    GenerativeModel,
};
use std::{error::Error, io::Write};
use tokio_stream::StreamExt;

// Get absolute path to example image
const IMAGE_PATH: &str = "examples/inline_data.jpg";

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
                Part::text("Convert the tabular data from the attached image into a JSON object and translate it to English if possible"),
                Part::image_from_path(IMAGE_PATH)?,
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
