use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, Part, Request, Tool},
    GenerativeModel,
};
use std::error::Error;

fn display_grounding_metadata(response: &gemini_ai_rust::models::Response) {
    for candidate in response.candidates.iter() {
        if let Some(metadata) = &candidate.grounding_metadata {
            // Display search queries used
            if let Some(queries) = &metadata.web_search_queries {
                println!("\n{}", "🔎 Search Queries Used:".blue().bold());
                for query in queries {
                    println!("   • {}", query.cyan());
                }
            }

            // Display grounding chunks (sources)
            if let Some(chunks) = &metadata.grounding_chunks {
                println!("\n{}", "📚 Sources:".yellow().bold());
                for (i, chunk) in chunks.iter().enumerate() {
                    if let Some(web) = &chunk.web {
                        println!(
                            "   {}. {}",
                            (i + 1).to_string().yellow(),
                            web.title
                                .as_ref()
                                .unwrap_or(&"Untitled".to_string())
                                .white()
                                .bold()
                        );
                        if let Some(uri) = &web.uri {
                            println!("      {}", uri.bright_black().italic());
                        }
                    }
                }
            }

            // Display grounding supports (evidence)
            if let Some(supports) = &metadata.grounding_supports {
                println!("\n{}", "🔍 Evidence:".green().bold());
                for (i, support) in supports.iter().enumerate() {
                    if let Some(segment) = &support.segment {
                        if let Some(text) = &segment.text {
                            println!("   {}. {}", (i + 1).to_string().green(), text.white());

                            // Display confidence scores
                            if let Some(scores) = &support.confidence_scores {
                                for score in scores {
                                    let confidence = format!("{:.1}%", score * 100.0);
                                    let colored_score = match score * 100.0 {
                                        x if x >= 90.0 => confidence.bright_green(),
                                        x if x >= 70.0 => confidence.green(),
                                        x if x >= 50.0 => confidence.yellow(),
                                        _ => confidence.red(),
                                    };
                                    println!("      Confidence: {}", colored_score);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Display the actual response text
        println!("\n{}", "💡 AI Response:".magenta().bold());
        for part in &candidate.content.parts {
            match part {
                Part::Text { text } => {
                    println!("{}", text.bright_white());
                }
                _ => println!("{}", "Unsupported content type".red()),
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🤖 Gemini AI Search Demo".bright_green().bold());
    println!("{}", "========================".bright_green());

    // Load environment variables from .env file
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-2.0-flash-exp")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Example search queries
    let queries = vec![
        // "What are the top 3 programming languages in 2024?",
        "What is the latest version of Rust and its key features?",
    ];

    for query in queries {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "🔍 Query:".blue().bold(), query);

        let request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: query.into() }],
            }])
            .tools(vec![Tool::GOOGLE_SEARCH])
            .build();

        match model.generate_response(request).await {
            Ok(response) => {
                display_grounding_metadata(&response);
            }
            Err(e) => {
                println!("{} {}", "❌ Error:".red().bold(), e);
            }
        }
    }

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
