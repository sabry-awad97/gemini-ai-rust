use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{CodeExecutionOutcome, Content, Part, Request, Tool},
    GenerativeModel,
};
use std::{error::Error, io::Write};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🤖 Gemini Code Execution Demo".bright_green().bold());
    println!("{}", "===========================".bright_green());

    // Load environment variables from .env file
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Example coding tasks
    let tasks = vec![
        "What is the sum of the first 50 prime numbers? Generate and run code for the calculation.",
        "Create a function to check if a string is a palindrome, then test it with some examples.",
    ];

    for task in tasks {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "🔍 Task:".blue().bold(), task);

        // Create a request with code execution enabled
        let request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: task.into() }],
            }])
            .tools(vec![Tool::CODE_EXECUTION])
            .build();

        // Stream the response
        let mut stream = model.stream_generate_response(request).await?;
        let mut stdout = std::io::stdout();

        // Track if we've seen certain parts to avoid duplicates
        let mut seen_code = false;
        let mut seen_result = false;

        while let Some(response) = stream.next().await {
            match response {
                Ok(response) => {
                    if let Some(candidates) = response.candidates {
                        for candidate in candidates {
                            if let Some(content) = candidate.content {
                                for part in content.parts {
                                    match part {
                                        Part::Text { text } => {
                                            print!("{}", text.bright_white());
                                            stdout.flush()?;
                                        }
                                        Part::ExecutableCode { executable_code } => {
                                            if !seen_code {
                                                println!(
                                                    "\n{}",
                                                    "📝 Generated Code:".yellow().bold()
                                                );
                                                println!(
                                                    "\n{} {}",
                                                    "Language:".bright_yellow(),
                                                    executable_code.language
                                                );
                                                println!(
                                                    "{}\n{}",
                                                    "Code:".bright_yellow(),
                                                    executable_code.code.cyan()
                                                );
                                                seen_code = true;
                                            }
                                        }
                                        Part::CodeExecutionResult {
                                            code_execution_result,
                                        } => {
                                            if !seen_result {
                                                println!(
                                                    "\n{}",
                                                    "🚀 Execution Results:".magenta().bold()
                                                );
                                                let outcome_str =
                                                    format!("{:?}", code_execution_result.outcome);
                                                let colored_outcome = match code_execution_result
                                                    .outcome
                                                {
                                                    CodeExecutionOutcome::Ok => outcome_str.green(),
                                                    _ => outcome_str.red(),
                                                };
                                                println!(
                                                    "\n{} {}",
                                                    "Outcome:".bright_magenta(),
                                                    colored_outcome
                                                );
                                                println!(
                                                    "{}\n{}",
                                                    "Output:".bright_magenta(),
                                                    code_execution_result.output.white()
                                                );
                                                seen_result = true;
                                            }
                                        }
                                        _ => {} // Ignore other part types
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{} {}", "❌ Error:".red().bold(), e);
                }
            }
        }
    }

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
