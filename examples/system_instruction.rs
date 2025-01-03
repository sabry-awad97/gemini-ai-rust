use dotenv::dotenv;
use futures::StreamExt;
use gemini_ai_rust::{
    models::{Content, Part, Request, Role},
    GenerativeModel,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Create a new model instance
    let model = GenerativeModel::from_env("gemini-1.5-pro")?;

    // Create a request with system instruction
    let request = Request::builder()
        .system_instruction(Some("You are a cat. Your name is Neko.".into()))
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::text("What's your name and what do you like to do?")],
        }])
        .build();

    // Generate a response
    println!("Generating response...");
    let mut stream = model.stream_generate_response(request).await?;
    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => print!("{}", response.text()),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    println!();

    Ok(())
}
