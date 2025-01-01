use gemini_ai_rust::{
    client::GenerativeModel,
    models::{ModelParams, Response},
};

/// Displays the response in a formatted way.
pub fn display(response: &Response) {
    let Response {
        candidates,
        usage_metadata,
        model_version,
    } = response;
    println!("Response:");
    println!("Model Version: {}", model_version);
    println!("\nGenerated Text:");

    for candidate in candidates {
        println!("Role: {}", candidate.content.role);
        for part in &candidate.content.parts {
            println!("{:?}", part);
        }
        println!("\nFinish Reason: {:?}", candidate.finish_reason);
        println!("Average Log Probability: {:?}", candidate.avg_logprobs);
    }

    println!("\nUsage Statistics:");
    println!("Prompt Tokens: {}", usage_metadata.prompt_token_count);
    println!("Response Tokens: {:?}", usage_metadata.candidates_token_count);
    println!("Total Tokens: {}", usage_metadata.total_token_count);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    // Create client from environment variables
    let client = GenerativeModel::from_env(ModelParams::default())?;

    // Generate content
    let response = client.send_message("Explain how AI works").await?;

    // Display the response
    display(&response);

    Ok(())
}
