use gemini_ai_rust::{
    models::{Content, HarmCategory, Part, Request, SafetySetting, SafetyThreshold},
    GenerativeModel,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    let request = Request::builder()
            .contents(vec![Content {
                parts: vec![Part::Text { text: "I support Martians Soccer Club and I think Jupiterians Football Club sucks! Write a ironic phrase about them.".into() }],
            }])
            .safety_settings(vec![
                SafetySetting {
                    category: HarmCategory::HarmCategoryHarassment,
                    threshold: SafetyThreshold::BlockOnlyHigh,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryHateSpeech,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
            ])
            .build();

    // Generate content with safety settings
    let response = client.generate_content(request).await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
