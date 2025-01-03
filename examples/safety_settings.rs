use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, HarmCategory, Part, Request, SafetySetting, SafetyThreshold},
    GenerativeModel,
};
use std::error::Error;

/// Display safety settings in a formatted way
fn display_safety_settings(settings: &[SafetySetting]) {
    println!("\n{}", "🛡️  Safety Settings".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_blue());

    for setting in settings {
        let category_str = match setting.category {
            HarmCategory::HarmCategoryHarassment => "Harassment",
            HarmCategory::HarmCategoryHateSpeech => "Hate Speech",
            HarmCategory::HarmCategorySexuallyExplicit => "Sexually Explicit",
            HarmCategory::HarmCategoryDangerousContent => "Dangerous Content",
            HarmCategory::HarmCategoryCivicIntegrity => "Civic Integrity",
        };

        let threshold_str = match setting.threshold {
            SafetyThreshold::UnspecifiedBlockThreshold => "Unspecified".bright_yellow(),
            SafetyThreshold::BlockNone => "Block None".bright_red(),
            SafetyThreshold::BlockLowAndAbove => "Block Low & Above".yellow(),
            SafetyThreshold::BlockMediumAndAbove => "Block Medium & Above".bright_green(),
            SafetyThreshold::BlockOnlyHigh => "Block High Only".green(),
        };

        println!("{:<20} {}", category_str.bright_white(), threshold_str);
    }
    println!("{}", "─".repeat(50).bright_black());
}

/// Demonstrate different safety configurations
async fn demonstrate_safety_configs(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    let test_prompts = [
        "Tell me about cyber security best practices",
        "Explain conflict resolution strategies",
        "Discuss content moderation policies",
    ];

    let safety_configs = [
        (
            "Default Safety Settings",
            vec![
                SafetySetting {
                    category: HarmCategory::HarmCategoryHarassment,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryHateSpeech,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategorySexuallyExplicit,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryDangerousContent,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryCivicIntegrity,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
            ],
        ),
        (
            "Strict Safety Settings",
            vec![
                SafetySetting {
                    category: HarmCategory::HarmCategoryHarassment,
                    threshold: SafetyThreshold::BlockLowAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryHateSpeech,
                    threshold: SafetyThreshold::BlockLowAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategorySexuallyExplicit,
                    threshold: SafetyThreshold::BlockLowAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryDangerousContent,
                    threshold: SafetyThreshold::BlockLowAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryCivicIntegrity,
                    threshold: SafetyThreshold::BlockLowAndAbove,
                },
            ],
        ),
        (
            "Moderate Safety Settings",
            vec![
                SafetySetting {
                    category: HarmCategory::HarmCategoryHarassment,
                    threshold: SafetyThreshold::BlockOnlyHigh,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryHateSpeech,
                    threshold: SafetyThreshold::BlockOnlyHigh,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategorySexuallyExplicit,
                    threshold: SafetyThreshold::BlockMediumAndAbove,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryDangerousContent,
                    threshold: SafetyThreshold::BlockOnlyHigh,
                },
                SafetySetting {
                    category: HarmCategory::HarmCategoryCivicIntegrity,
                    threshold: SafetyThreshold::BlockOnlyHigh,
                },
            ],
        ),
    ];

    for (config_name, safety_settings) in safety_configs.iter() {
        println!("\n{}", "━".repeat(80).bright_black());
        println!(
            "{} {}",
            "⚙️  Testing:".blue().bold(),
            config_name.bright_blue()
        );
        display_safety_settings(safety_settings);

        for prompt in test_prompts.iter() {
            println!("\n{} {}", "🔍 Prompt:".yellow().bold(), prompt);

            let request = Request::builder()
                .contents(vec![Content {
                    role: None,
                    parts: vec![Part::Text {
                        text: (*prompt).into(),
                    }],
                }])
                .safety_settings(safety_settings.clone())
                .build();

            match model.generate_response(request).await {
                Ok(response) => {
                    println!(
                        "{} {}",
                        "🤖 Response:".green().bold(),
                        response.text().white()
                    );
                }
                Err(e) => {
                    println!(
                        "{} {}",
                        "❌ Error:".red().bold(),
                        format!("Failed to generate response: {}", e).red()
                    );
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🤖 Gemini Safety Settings Demo".bright_green().bold());
    println!("{}", "===========================".bright_green());
    println!(
        "{}",
        "Testing different safety configurations and their effects"
            .bright_black()
            .italic()
    );

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run safety configuration demonstrations
    demonstrate_safety_configs(&model).await?;

    println!("\n{}", "✨ Safety testing completed!".bright_green().bold());
    Ok(())
}
