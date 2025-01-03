use colored::*;
use dotenv::dotenv;
use futures::StreamExt;
use gemini_ai_rust::{
    models::{Content, Part, Request},
    GenerativeModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{error::Error, time::Duration};

async fn demonstrate_streaming(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    let prompts = [
        (
            "Creative Writing",
            "Write a short story about a robot learning to paint",
        ),
        (
            "Technical Explanation",
            "Explain how quantum computing works, step by step",
        ),
        (
            "Problem Solving",
            "Describe approaches to optimize a slow database query",
        ),
    ];

    for (category, prompt) in prompts {
        println!("\n{}", "━".repeat(80).bright_black());
        println!(
            "{} {}",
            "📝 Category:".blue().bold(),
            category.bright_blue()
        );
        println!(
            "{} {}",
            "🔍 Prompt:".yellow().bold(),
            prompt.bright_yellow()
        );

        // Create a fancy spinner
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈")
                .template("{spinner:.green} {msg}")?,
        );
        pb.set_message("Generating response...");
        pb.enable_steady_tick(Duration::from_millis(100));

        let request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text {
                    text: prompt.into(),
                }],
            }])
            .build();

        let mut stream = model.stream_generate_response(request).await?;
        pb.finish_and_clear();

        println!("{}", "🤖 Response:".green().bold());
        println!("{}", "─".repeat(80).bright_black());

        let mut first_chunk = true;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(response) => {
                    if !first_chunk {
                        // Small delay between chunks for better readability
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    print!("{}", response.text().white());
                    first_chunk = false;
                }
                Err(e) => {
                    println!(
                        "\n{} {}",
                        "❌ Error:".red().bold(),
                        format!("Failed to get response chunk: {}", e).red()
                    );
                    break;
                }
            }
        }
        println!("\n{}", "─".repeat(80).bright_black());
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🚀 Gemini Streaming Demo".bright_green().bold());
    println!("{}", "=======================".bright_green());
    println!(
        "{}",
        "Demonstrating real-time streaming responses"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Initialize the model
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run streaming demonstrations
    demonstrate_streaming(&model).await?;

    println!("\n{}", "✨ Streaming demo completed!".bright_green().bold());
    Ok(())
}
