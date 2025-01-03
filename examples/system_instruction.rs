use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Role},
    GenerativeModel,
};
use std::error::Error;

async fn demonstrate_personas(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    let personas = [
        (
            "Professional Translator",
            "You are a professional translator specializing in English to French translation. \
             Maintain formal language and provide pronunciation guides when relevant.",
            "Translate the phrase: 'Welcome to our international conference'",
        ),
        (
            "Technical Expert",
            "You are a senior software architect with 15 years of experience in distributed systems. \
             Provide detailed technical explanations with code examples when appropriate.",
            "Explain the difference between eventual consistency and strong consistency",
        ),
        (
            "Creative Writer",
            "You are an award-winning novelist known for vivid descriptions and emotional depth. \
             Write in a engaging and descriptive style.",
            "Describe a sunset over a bustling city",
        ),
        (
            "Data Analyst",
            "You are a data scientist specializing in explaining complex statistics in simple terms. \
             Use analogies and real-world examples in your explanations.",
            "Explain what p-value means in statistics",
        ),
    ];

    for (persona_name, system_instruction, prompt) in personas {
        println!("\n{}", "━".repeat(100).bright_black());
        println!(
            "{} {}",
            "👤 Persona:".blue().bold(),
            persona_name.bright_blue()
        );
        println!(
            "{} {}",
            "🎯 System Instruction:".yellow().bold(),
            system_instruction.bright_yellow()
        );
        println!("{} {}", "❓ Prompt:".cyan().bold(), prompt.bright_cyan());
        println!("{}", "─".repeat(100).bright_black());

        let request = Request::builder()
            .system_instruction(Some(system_instruction.into()))
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: prompt.into(),
                }],
            }])
            .build();

        match model.generate_response(request).await {
            Ok(response) => {
                println!("{}", "🤖 Response:".green().bold());
                println!("{}", response.text().white());
            }
            Err(e) => {
                println!(
                    "{} {}",
                    "❌ Error:".red().bold(),
                    format!("Failed to generate response: {}", e).red()
                );
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🎭 Gemini Personas Demo".bright_green().bold());
    println!("{}", "=====================".bright_green());
    println!(
        "{}",
        "Demonstrating different AI personas through system instructions"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Initialize the model
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run persona demonstrations
    demonstrate_personas(&model).await?;

    println!("\n{}", "✨ Personas demo completed!".bright_green().bold());
    Ok(())
}
