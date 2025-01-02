use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Role},
    GenerativeModel,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Create a request with some text
    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "The quick brown fox jumps over the lazy dog.".into(),
            }],
        }])
        .build();

    // Count tokens in the text
    let token_count = model.count_tokens(request).await?;
    println!("Total tokens: {}", token_count.total_tokens);

    Ok(())
}
