use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    models::{Content, GenerationConfig, HarmCategory, Part, Request, SafetyThreshold},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Prepare the request
    let request = Request::builder()
        .contents(vec![Content {
            parts: vec![Part::Text {
                text: "Write a story about a magic backpack.".into(),
            }],
        }])
        .safety_settings([(
            HarmCategory::HarmCategoryDangerousContent,
            SafetyThreshold::BlockOnlyHigh,
        )
            .into()])
        .generation_config(
            GenerationConfig::builder()
                .stop_sequences(vec!["Title".to_string()])
                .temperature(1.0)
                .max_output_tokens(800)
                .top_p(0.8)
                .top_k(10)
                .build(),
        )
        .build();

    let response = client.generate_content(request).await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
