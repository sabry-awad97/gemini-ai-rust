use colored::*;
use futures::StreamExt;
use gemini_ai_rust::{chat::ChatSession, GenerativeModel};
use std::{error::Error, io::Write};

async fn run_regular_chat(model: GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "💬 Regular Chat Demo".bright_blue().bold());
    println!("{}", "=================".bright_blue());

    // Create a chat session with a system instruction
    let mut chat = ChatSession::new(model).with_system_instruction(
        "You are a helpful AI assistant with expertise in Rust programming. \
        You provide clear, concise answers with code examples when relevant. \
        You format your code examples with proper syntax highlighting.",
    );

    println!(
        "{}",
        "✓ Chat session initialized with Rust programming expertise".green()
    );

    // Example conversation about Rust programming
    let messages = [
        "What are the key features that make Rust unique compared to other programming languages?",
        "Can you show me an example of using Option and Result types in Rust?",
        "How can I implement error handling for this code?",
    ];

    for message in messages {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "👤 User:".blue().bold(), message);
        print!("{} ", "🤖 Assistant:".green().bold());
        std::io::stdout().flush()?;

        let response = chat.send_message(message).await?;
        println!("{}", response.white());
    }

    Ok(())
}

async fn run_streaming_chat(model: GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "📡 Streaming Chat Demo".bright_magenta().bold());
    println!("{}", "====================".bright_magenta());

    // Create a new chat session for streaming with creative storytelling
    let mut streaming_chat = ChatSession::new(model).with_system_instruction(
        "You are a creative storyteller who crafts engaging short stories. \
        Your stories are imaginative and incorporate modern technology themes. \
        You break your stories into small paragraphs for better readability.",
    );

    println!(
        "{}",
        "✓ Streaming chat session initialized with storytelling capabilities".green()
    );

    // Creative writing prompts
    let prompts = [
        "Tell me a short story about a programmer who discovers their AI assistant has become sentient.",
        "Write a story about a time traveler who visits the first computer ever built.",
    ];

    for prompt in prompts {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "👤 User:".blue().bold(), prompt);
        print!("{} ", "🤖 Assistant:".green().bold());
        std::io::stdout().flush()?;

        // Start streaming chat
        let mut stream = streaming_chat.send_message_streaming(prompt).await?;

        // Print the streaming responses
        while let Some(response) = stream.next().await {
            match response {
                Ok(response) => {
                    print!("{}", response.text().white());
                    std::io::stdout().flush()?;
                }
                Err(e) => {
                    eprintln!("{} {}", "❌ Error:".red().bold(), e);
                }
            }
        }
        println!("\n");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🤖 Gemini Chat Demo".bright_green().bold());
    println!("{}", "=================".bright_green());

    // Load environment variables
    dotenv::dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run both chat examples
    run_regular_chat(model.clone()).await?;
    run_streaming_chat(model).await?;

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
