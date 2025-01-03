use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::client::GenerativeModel;
use std::error::Error;

/// Display model information in a formatted way
fn display_model_info(model_name: &str, model: &gemini_ai_rust::models::ModelInfo) {
    println!("\n{}", "━".repeat(80).bright_black());
    println!("{} {}", "🤖 Name:".blue().bold(), model_name.bright_blue());
    println!(
        "{} {}",
        "📋 Display Name:".cyan().bold(),
        model.display_name.bright_cyan()
    );
    println!(
        "{} {}",
        "📝 Description:".yellow().bold(),
        model.description.bright_yellow()
    );
    println!(
        "{} {}",
        "🔢 Version:".magenta().bold(),
        model.version.bright_magenta()
    );

    // Token limits
    println!("\n{}", "📊 Token Limits:".green().bold());
    println!(
        "   {:<20} {}",
        "Input Limit:".white(),
        model.input_token_limit.to_string().bright_green()
    );
    println!(
        "   {:<20} {}",
        "Output Limit:".white(),
        model.output_token_limit.to_string().bright_green()
    );

    // Generation methods
    println!(
        "\n{} {}",
        "⚡ Supported Methods:".red().bold(),
        model.supported_generation_methods.join(", ").bright_red()
    );

    // Model parameters
    println!("\n{}", "🎛️  Default Parameters:".bright_white().bold());
    if let Some(temp) = model.temperature {
        println!(
            "   {:<20} {}",
            "Temperature:".white(),
            temp.to_string().bright_white()
        );
    }
    if let Some(top_p) = model.top_p {
        println!(
            "   {:<20} {}",
            "Top P:".white(),
            top_p.to_string().bright_white()
        );
    }
    if let Some(top_k) = model.top_k {
        println!(
            "   {:<20} {}",
            "Top K:".white(),
            top_k.to_string().bright_white()
        );
    }
}

/// Get and display details for a specific model
async fn display_model_details(
    model: &GenerativeModel,
    model_name: &str,
) -> Result<(), Box<dyn Error>> {
    println!(
        "\n{}\n{} {}",
        "Getting details for:".bright_black().italic(),
        "🔍".blue(),
        model_name.bright_blue().bold()
    );

    match model.get_model_info(model_name).await {
        Ok(model_info) => display_model_info(model_name, &model_info),
        Err(e) => println!(
            "{} {}",
            "❌ Error:".red().bold(),
            format!("Failed to get model info: {}", e).red()
        ),
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🌟 Gemini Models Explorer".bright_green().bold());
    println!("{}", "======================".bright_green());
    println!(
        "{}",
        "Discovering available Gemini AI models"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Initialize the model
    let model = GenerativeModel::from_env("gemini-ai-rust")?;
    println!("{}", "✓ Gemini client initialized".green());

    // List all available models
    println!(
        "\n{}",
        "📚 Listing all available models:".bright_blue().bold()
    );
    match model.list_models().await {
        Ok(models) => {
            for model_info in &models.models {
                display_model_info(&model_info.name, model_info);
            }
        }
        Err(e) => {
            println!(
                "{} {}",
                "❌ Error:".red().bold(),
                format!("Failed to list models: {}", e).red()
            );
            return Err(e.into());
        }
    }

    // Get details for specific models
    let featured_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision"];

    println!(
        "\n{}",
        "🔍 Featured Models Details:".bright_magenta().bold()
    );
    for model_name in featured_models {
        display_model_details(&model, model_name).await?;
    }

    println!(
        "\n{}",
        "✨ Model exploration completed!".bright_green().bold()
    );
    Ok(())
}
