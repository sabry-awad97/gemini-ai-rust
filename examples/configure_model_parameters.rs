use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    models::{Content, GenerationConfig, HarmCategory, Part, Request, Role, SafetyThreshold},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Prepare the request
    let var_name = vec![Content {
        role: Some(Role::User),
        parts: vec![Part::Text {
            text: "Write a story about a magic backpack.".into(),
        }],
    }];
    let request = Request::builder()
        .contents(var_name)
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

    let response = client.generate_response(request).await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
