use gemini_ai_rust::{
    models::{HarmCategory, SafetySetting, SafetyThreshold},
    GenerativeModel,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Create safety settings
    let safety_settings = vec![
        SafetySetting {
            category: HarmCategory::HarmCategoryHarassment,
            threshold: SafetyThreshold::BlockOnlyHigh,
        },
        SafetySetting {
            category: HarmCategory::HarmCategoryHateSpeech,
            threshold: SafetyThreshold::BlockMediumAndAbove,
        },
    ];

    // Generate content with safety settings
    let response = client
        .generate_content_with_safety(
            "I support Martians Soccer Club and I think Jupiterians Football Club sucks! Write a ironic phrase about them.",
            safety_settings,
        )
        .await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
