use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request},
    GenerativeModel,
};
use std::error::Error;

/// Demonstrate token counting for different types of content
async fn demonstrate_token_counting(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    let test_cases = [
        (
            "Short Text",
            "Hello, how are you today?",
            "Simple greeting message",
        ),
        (
            "Medium Text",
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once.",
            "Classic pangram with additional context",
        ),
        (
            "Technical Text",
            "In computer programming, a token is a string of characters, categorized according to the rules of the programming language, that serves as the smallest unit of meaning.",
            "Technical definition with specialized terms",
        ),
        (
            "Multi-language Text",
            "Hello World! Bonjour le monde! ¡Hola Mundo! Hallo Welt! Ciao Mondo! 你好世界！",
            "Greetings in multiple languages",
        ),
        (
            "Code Snippet",
            r#"fn main() {
    println!("Hello, World!");
    let x = 42;
    let message = format!("The answer is {}", x);
}"#,
            "Rust code example",
        ),
    ];

    for (case_name, text, description) in test_cases {
        println!("\n{}", "━".repeat(100).bright_black());
        println!(
            "{} {}",
            "📝 Test Case:".blue().bold(),
            case_name.bright_blue()
        );
        println!(
            "{} {}",
            "📋 Description:".cyan().bold(),
            description.bright_cyan()
        );
        println!("{}", "─".repeat(100).bright_black());

        // Display input text with stats
        println!("{}", "📥 Input Text:".yellow().bold());
        println!("{}", text.bright_white());
        println!(
            "{} {}",
            "📊 Text Stats:".magenta().bold(),
            format!(
                "{} characters, {} words",
                text.chars().count(),
                text.split_whitespace().count()
            )
            .bright_magenta()
        );

        // Count tokens
        let request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: text.into() }],
            }])
            .build();

        match model.count_tokens(request).await {
            Ok(token_count) => {
                println!(
                    "\n{} {}",
                    "🔢 Token Count:".green().bold(),
                    token_count.total_tokens.to_string().bright_green()
                );

                // Calculate token ratios
                let chars_per_token = text.chars().count() as f64 / token_count.total_tokens as f64;
                let words_per_token =
                    text.split_whitespace().count() as f64 / token_count.total_tokens as f64;

                println!("{}", "📈 Token Analysis:".bright_white().bold());
                println!(
                    "   {:<25} {:.2}",
                    "Characters per token:".white(),
                    chars_per_token
                );
                println!(
                    "   {:<25} {:.2}",
                    "Words per token:".white(),
                    words_per_token
                );
            }
            Err(e) => {
                println!(
                    "{} {}",
                    "❌ Error:".red().bold(),
                    format!("Failed to count tokens: {}", e).red()
                );
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🔢 Gemini Token Counter".bright_green().bold());
    println!("{}", "====================".bright_green());
    println!(
        "{}",
        "Analyzing token counts for different types of text"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Initialize the model
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run token counting demonstrations
    demonstrate_token_counting(&model).await?;

    println!("\n{}", "✨ Token analysis completed!".bright_green().bold());
    Ok(())
}
