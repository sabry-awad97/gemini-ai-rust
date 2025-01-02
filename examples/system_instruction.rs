use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Role},
    GenerativeModel,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    let request = Request::builder()
        .system_instruction(Content {
            role: None,
            parts: vec![Part::Text {
                text: "You are a cat. Your name is Neko.".into(),
            }],
        })
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "Hello there".into(),
            }],
        }])
        .build();

    // Generate content with system instruction
    let response = client.generate_response(request).await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
