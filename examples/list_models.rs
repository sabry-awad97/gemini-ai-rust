use dotenv::dotenv;
use gemini_ai_rust::client::GenerativeModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a model instance
    let model = GenerativeModel::from_env("gemini-ai-rust")?;

    println!("Listing all available models:");
    println!("-----------------------------");

    // List all models
    let models = model.list_models().await?;
    for model_info in &models.models {
        println!("Name: {}", model_info.name);
        println!("Display name: {}", model_info.display_name);
        println!("Description: {}", model_info.description);
        println!("Version: {}", model_info.version);
        println!("Input token limit: {}", model_info.input_token_limit);
        println!("Output token limit: {}", model_info.output_token_limit);
        println!(
            "Supported generation methods: {}",
            model_info.supported_generation_methods.join(", ")
        );
        if let Some(temp) = model_info.temperature {
            println!("Default temperature: {}", temp);
        }
        if let Some(max_temp) = model_info.max_temperature {
            println!("Max temperature: {}", max_temp);
        }
        if let Some(top_p) = model_info.top_p {
            println!("Default top_p: {}", top_p);
        }
        if let Some(top_k) = model_info.top_k {
            println!("Default top_k: {}", top_k);
        }
        println!("-----------------------------");
    }

    println!("\nGetting details for gemini-1.5-flash:");
    println!("-----------------------------");

    // Get specific model details
    let model_info = model.get_model("gemini-1.5-flash").await?;
    println!("Name: {}", model_info.name);
    println!("Display name: {}", model_info.display_name);
    println!("Description: {}", model_info.description);
    println!("Version: {}", model_info.version);
    println!("Input token limit: {}", model_info.input_token_limit);
    println!("Output token limit: {}", model_info.output_token_limit);
    println!(
        "Supported generation methods: {}",
        model_info.supported_generation_methods.join(", ")
    );
    if let Some(temp) = model_info.temperature {
        println!("Default temperature: {}", temp);
    }
    if let Some(max_temp) = model_info.max_temperature {
        println!("Max temperature: {}", max_temp);
    }
    if let Some(top_p) = model_info.top_p {
        println!("Default top_p: {}", top_p);
    }
    if let Some(top_k) = model_info.top_k {
        println!("Default top_k: {}", top_k);
    }

    Ok(())
}
