use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, GenerationConfig, HarmCategory, Part, Request, Role, SafetyThreshold},
    GenerativeModel,
};
use std::error::Error;

/// Demonstrates model behavior with different temperature settings
async fn demonstrate_temperature(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🌡️  Temperature Demo".bright_blue().bold());
    println!("{}", "=================".bright_blue());
    println!(
        "{}",
        "Temperature controls response creativity (0.0 = focused, 1.0 = creative)".bright_black()
    );

    let prompt = "Write a one-sentence story about a robot.";
    let temperatures = [0.0, 0.5, 1.0];

    for temp in temperatures {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "🎯 Temperature:".yellow().bold(), temp);

        let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: prompt.into(),
                }],
            }])
            .generation_config(
                GenerationConfig::builder()
                    .temperature(temp)
                    .max_output_tokens(50)
                    .build(),
            )
            .build();

        let response = model.generate_response(request).await?;
        println!(
            "{} {}",
            "🤖 Response:".green().bold(),
            response.text().white()
        );
    }

    Ok(())
}

/// Demonstrates safety settings with different thresholds
async fn demonstrate_safety_settings(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🛡️  Safety Settings Demo".bright_magenta().bold());
    println!("{}", "=====================".bright_magenta());
    println!(
        "{}",
        "Safety settings control content filtering levels".bright_black()
    );

    let prompt = "Write about a conflict between two people.";
    let safety_levels = [
        (SafetyThreshold::BlockNone, "No Blocking"),
        (SafetyThreshold::BlockOnlyHigh, "Block High"),
        (SafetyThreshold::BlockMediumAndAbove, "Block Medium+"),
    ];

    for (threshold, name) in safety_levels {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "🔒 Safety Level:".yellow().bold(), name);

        let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: prompt.into(),
                }],
            }])
            .safety_settings([(HarmCategory::HarmCategoryHarassment, threshold).into()])
            .build();

        let response = model.generate_response(request).await?;
        println!(
            "{} {}",
            "🤖 Response:".green().bold(),
            response.text().white()
        );
    }

    Ok(())
}

/// Demonstrates top_k and top_p sampling parameters
async fn demonstrate_sampling(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🎲 Sampling Parameters Demo".bright_cyan().bold());
    println!("{}", "=======================".bright_cyan());
    println!(
        "{}",
        "Top-k and Top-p control token selection diversity".bright_black()
    );

    let prompt = "List 3 creative names for a tech startup.";
    let configs = [
        (5, 0.5, "Focused"),
        (20, 0.8, "Balanced"),
        (40, 0.95, "Diverse"),
    ];

    for (top_k, top_p, style) in configs {
        println!("\n{}", "━".repeat(50).bright_black());
        println!(
            "{} {} (top_k={}, top_p={})",
            "🎯 Style:".yellow().bold(),
            style,
            top_k,
            top_p
        );

        let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: prompt.into(),
                }],
            }])
            .generation_config(
                GenerationConfig::builder()
                    .temperature(0.9)
                    .top_k(top_k)
                    .top_p(top_p)
                    .max_output_tokens(100)
                    .build(),
            )
            .build();

        let response = model.generate_response(request).await?;
        println!(
            "{} {}",
            "🤖 Response:".green().bold(),
            response.text().white()
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "{}",
        "🤖 Gemini Model Parameters Demo".bright_green().bold()
    );
    println!("{}", "==========================".bright_green());

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run demonstrations
    demonstrate_temperature(&model).await?;
    demonstrate_safety_settings(&model).await?;
    demonstrate_sampling(&model).await?;

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
