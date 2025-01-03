use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Tool},
    GenerativeModel,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "Google Search Example".bright_green().bold());
    println!("{}", "===================".bright_green());

    // Load environment variables from .env file
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-2.0-flash-exp")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Example search queries
    let queries = vec![
        "What are the top 3 programming languages in 2024?",
        "What is the latest version of Rust and its key features?",
    ];

    for query in queries {
        println!("\n{} {}", "🔍 Query:".blue().bold(), query);

        // Create a request with code execution enabled
        let request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: query.into() }],
            }])
            .tools(vec![Tool::GOOGLE_SEARCH])
            .build();

        // Get the response
        let response = model.generate_response(request).await?;

        println!("{}", "\n📝 Response:".yellow().bold());
        for candidate in response.candidates.iter() {
            for part in &candidate.content.parts {
                match part {
                    Part::Text { text } => {
                        println!("{}", text.cyan());
                    }
                    _ => println!("{}", "Unsupported content type".red()),
                }
            }
        }
        println!("{}", "---".bright_black());
    }

    println!("\n{}", "✨ Example completed successfully!".green().bold());
    Ok(())
}
