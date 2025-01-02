use gemini_ai_rust::{
    models::{Content, HarmCategory, Part, Request, Role, SafetySetting, SafetyThreshold},
    GenerativeModel,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
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
    let response = client.generate_response(request).await?;

    // Display the response
    println!("{}", response.text());

    Ok(())
}
