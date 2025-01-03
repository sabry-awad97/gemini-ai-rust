use colored::*;
use gemini_ai_rust::{
    models::{Content, Part, Request, Response},
    GenerativeModel,
};
use std::error::Error;

/// Displays the response in a formatted way.
fn display_response(response: &Response) {
    println!("\n{}", "📝 Response Details:".bright_blue().bold());

    if let Some(version) = &response.model_version {
        println!("{} {}", "Model Version:".blue(), version);
    }

    if let Some(candidates) = &response.candidates {
        for (i, candidate) in candidates.iter().enumerate() {
            println!(
                "\n{} {}",
                "✨ Candidate".yellow().bold(),
                (i + 1).to_string().yellow()
            );

            if let Some(content) = &candidate.content {
                if let Some(role) = &content.role {
                    println!("{} {:?}", "Role:".bright_yellow(), role);
                }

                println!("\n{}", "💭 Generated Text:".cyan().bold());
                for part in &content.parts {
                    match part {
                        Part::Text { text } => println!("{}", text.white()),
                        _ => println!("{}", "Unsupported content type".red()),
                    }
                }
            }

            if let Some(reason) = &candidate.finish_reason {
                println!("\n{} {:?}", "Finish Reason:".bright_yellow(), reason);
            }

            if let Some(prob) = &candidate.avg_logprobs {
                println!("{} {:.4}", "Confidence Score:".bright_yellow(), prob);
            }
        }
    }

    if let Some(usage) = &response.usage_metadata {
        println!("\n{}", "📊 Usage Statistics:".green().bold());
        println!(
            "{} {}",
            "Prompt Tokens:".bright_green(),
            usage.prompt_token_count
        );
        if let Some(resp_tokens) = usage.candidates_token_count {
            println!("{} {}", "Response Tokens:".bright_green(), resp_tokens);
        }
        println!(
            "{} {}",
            "Total Tokens:".bright_green(),
            usage.total_token_count
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🤖 Gemini Basic Demo".bright_green().bold());
    println!("{}", "==================".bright_green());

    // Load environment variables
    dotenv::dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Example prompts to demonstrate different capabilities
    let prompts = vec![
        "Explain how artificial intelligence works in simple terms.",
        "Write a haiku about programming.",
        "What are three interesting facts about space exploration?",
    ];

    for prompt in prompts {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "🔍 Prompt:".blue().bold(), prompt);

        // Create the request
        let request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text {
                    text: prompt.into(),
                }],
            }])
            .build();

        // Generate content
        match model.generate_response(request).await {
            Ok(response) => {
                display_response(&response);
            }
            Err(e) => {
                eprintln!("{} {}", "❌ Error:".red().bold(), e);
            }
        }
    }

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
