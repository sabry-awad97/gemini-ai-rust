use std::io::Write;

use dotenv::dotenv;
use futures::StreamExt;
use gemini_ai_rust::{
    client::GenerativeModel,
    models::{Content, Part, Request, Role},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Prepare the request
    let request = Request::builder()
        .system_instruction(Some(
            "You are a helpful assistant that translates English to German.".into(),
        ))
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "How are you?".into(),
            }],
        }])
        .build();

    // Stream the response
    let mut stream = client.stream_generate_response(request).await?;

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
