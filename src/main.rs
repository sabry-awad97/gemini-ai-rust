use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const MODEL: &str = "gemini-1.5-flash";

#[derive(Debug, Serialize)]
struct Request {
    contents: Vec<Content>,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Part {
    text: String,
}

#[derive(Debug, Deserialize)]
struct Response {
    candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: UsageMetadata,
    #[serde(rename = "modelVersion")]
    model_version: String,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: CandidateContent,
    #[serde(rename = "finishReason")]
    finish_reason: String,
    #[serde(rename = "avgLogprobs")]
    avg_logprobs: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateContent {
    parts: Vec<Part>,
    role: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: i32,
    candidates_token_count: i32,
    total_token_count: i32,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let api_key = env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY must be set");

    let client = reqwest::Client::new();
    let request = Request {
        contents: vec![Content {
            parts: vec![Part {
                text: "Explain how AI works".to_string(),
            }],
        }],
    };

    let url = format!(
        "{}/models/{}:generateContent?key={}",
        BASE_URL, MODEL, api_key
    );

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await?
        .json::<Response>()
        .await?;

    println!("Response:");
    println!("Model Version: {}", response.model_version);
    println!("\nGenerated Text:");
    for candidate in response.candidates {
        println!("Role: {}", candidate.content.role);
        for part in candidate.content.parts {
            println!("{}", part.text);
        }
        println!("\nFinish Reason: {}", candidate.finish_reason);
        println!("Average Log Probability: {}", candidate.avg_logprobs);
    }

    println!("\nUsage Statistics:");
    println!(
        "Prompt Tokens: {}",
        response.usage_metadata.prompt_token_count
    );
    println!(
        "Response Tokens: {}",
        response.usage_metadata.candidates_token_count
    );
    println!(
        "Total Tokens: {}",
        response.usage_metadata.total_token_count
    );

    Ok(())
}
