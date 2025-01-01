use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, ModelParams, Part, Request},
    GenerativeModel,
};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env(ModelParams::default())?;

    let request = Request::builder()
        .system_instruction(Content {
            parts: vec![Part::Text {
                text: "You are a cat. Your name is Neko.".into(),
            }],
        })
        .contents(vec![Content {
            parts: vec![Part::Text {
                text: "Hello there".into(),
            }],
        }])
        .build();

    // Generate content with system instruction
    let response = client.generate_content(request).await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
