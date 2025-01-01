use dotenv::dotenv;
use gemini_ai_rust::GenerativeModel;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Generate content with system instruction
    let response = client
        .generate_content_with_system("You are a cat. Your name is Neko.", "Hello there")
        .await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
