use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Tool},
    GenerativeModel,
};
use std::{error::Error, io::Write};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;

    println!("=== Code Execution Example ===\n");
    println!("Asking about prime numbers...\n");

    // Create a request with code execution enabled
    let request = Request::builder()
        .contents(vec![Content {
            role: None,
            parts: vec![Part::Text {
                text: "What is the sum of the first 50 prime numbers? Generate and run code for the calculation, and make sure you get all 50.".into(),
            }],
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
                // Process each candidate's parts
                for candidate in response.candidates {
                    for part in candidate.content.parts {
                        match part {
                            Part::Text { text } => {
                                print!("{}", text);
                                stdout.flush()?;
                            }
                            Part::ExecutableCode { executable_code } => {
                                if !seen_code {
                                    println!("\nGenerated Code:");
                                    println!("\nLanguage: {}", executable_code.language);
                                    println!("Code:\n{}", executable_code.code);
                                    seen_code = true;
                                }
                            }
                            Part::CodeExecutionResult {
                                code_execution_result,
                            } => {
                                if !seen_result {
                                    println!("\nExecution Results:");
                                    println!("\nOutcome: {:?}", code_execution_result.outcome);
                                    println!("Output:\n{}", code_execution_result.output);
                                    seen_result = true;
                                }
                            }
                            _ => {} // Ignore other part types
                        }
                    }
                }
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    Ok(())
}
