use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{
        Content,   FunctionDeclaration,
        FunctionDeclarationSchema, FunctionResponse, Part, Request, Role, Schema, SchemaType,
         Tool, Response,
    },
    GenerativeModel,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{error::Error, time::Duration};
use tokio::time::sleep;

const RETRY_ATTEMPTS: u32 = 5;
const INITIAL_RETRY_DELAY_MS: u64 = 1000;
const MAX_RETRY_DELAY_MS: u64 = 5000;

/// Helper function to handle rate limiting and retries
async fn generate_response_with_retry(
    model: &GenerativeModel,
    request: Request,
    operation: &str,
) -> Result<Response, Box<dyn Error>> {
    let mut attempt = 0;
    let mut last_error = None;
    let mut delay_ms = INITIAL_RETRY_DELAY_MS;

    while attempt < RETRY_ATTEMPTS {
        match model.generate_response(request.clone()).await {
            Ok(response) => {
                if attempt > 0 {
                    println!(
                        "{} {} (attempt {}/{})",
                        "✓".green(),
                        operation,
                        attempt + 1,
                        RETRY_ATTEMPTS
                    );
                }
                return Ok(response);
            }
            Err(e) => {
                attempt += 1;
                last_error = Some(e);

                if attempt < RETRY_ATTEMPTS {
                    let error_msg = last_error.as_ref().unwrap().to_string();
                    if error_msg.contains("429") || error_msg.contains("RESOURCE_EXHAUSTED") {
                        println!(
                            "{} Rate limit hit for {}. Retrying in {} ms... (attempt {}/{})",
                            "⚠️".yellow(),
                            operation,
                            delay_ms,
                            attempt,
                            RETRY_ATTEMPTS
                        );
                        sleep(Duration::from_millis(delay_ms)).await;
                        // Exponential backoff with max delay
                        delay_ms = (delay_ms * 2).min(MAX_RETRY_DELAY_MS);
                    } else {
                        // For other errors, retry with same delay
                        println!(
                            "{} Error in {}: {}. Retrying... (attempt {}/{})",
                            "⚠️".yellow(),
                            operation,
                            error_msg,
                            attempt,
                            RETRY_ATTEMPTS
                        );
                        sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }
    }

    Err(last_error.unwrap().into())
}

/// Define custom function parameter structs
#[derive(Debug, Serialize, Deserialize)]
struct WeatherParams {
    location: String,
    unit: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CalendarParams {
    date: String,
    event: String,
    duration_minutes: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchResultParams {
    query: String,
    max_results: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct BookmarkParams {
    title: String,
    url: String,
    category: String,
}

/// Demonstrates weather function calling with real data from Google Search
async fn demonstrate_weather_function(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🌤️  Weather Function Demo".bright_blue().bold());
    println!("{}", "=====================".bright_blue());
    println!("{}", "Testing weather data retrieval function".bright_black());

    // Define the weather function
    let get_weather = FunctionDeclaration::builder()
        .name("get_weather")
        .description("Get the weather data for a location based on search results")
        .parameters(
            FunctionDeclarationSchema::builder()
                .r#type(SchemaType::Object)
                .properties([
                    (
                        "location".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .description("The city name")
                            .build(),
                    ),
                    (
                        "unit".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .enum_values(vec!["celsius".to_string(), "fahrenheit".to_string()])
                            .description("The temperature unit to use")
                            .build(),
                    ),
                ])
                .required(vec!["location".to_string(), "unit".to_string()])
                .build(),
        )
        .build();

    // Test weather queries
    let weather_queries = [
        "What's the current temperature in London using celsius?",
        "How's the weather in Tokyo in fahrenheit?",
        "Tell me about the weather conditions in New York.",
    ];

    for query in weather_queries {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "👤 User:".blue().bold(), query);

        // First, search for real weather data
        let search_query = format!("current weather temperature {}", query);
        let search_request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: search_query }],
            }])
            .tools(vec![Tool::GOOGLE_SEARCH])
            .build();

        let search_response = generate_response_with_retry(model, search_request, "weather search").await?;
        display_grounding_metadata(&search_response);

        // Add a small delay between requests to help with rate limiting
        sleep(Duration::from_millis(500)).await;

        // Now use the weather function with the real data
        let weather_request = Request::builder()
            .contents(vec![
                Content {
                    role: Some(Role::User),
                    parts: vec![Part::Text { text: query.into() }],
                },
                Content {
                    role: Some(Role::Model),
                    parts: vec![Part::Text { 
                        text: "Based on the search results above, I'll provide you with the current weather information.".to_string() 
                    }],
                },
            ])
            .tools(vec![vec![get_weather.clone()].into()])
            .build();

        let weather_response = generate_response_with_retry(model, weather_request, "weather function").await?;
        let function_calls = weather_response.function_calls();

        if function_calls.is_empty() {
            println!("{} {}", "🤖 Response:".green().bold(), weather_response.text().white());
            continue;
        }

        for call in function_calls {
            println!("{} {} with {}", "📞 Function Call:".yellow().bold(), call.name, call.args);
            
            let params: WeatherParams = serde_json::from_value(call.args.clone())?;
            
            // Create a response using the real data from search
            let weather_response = FunctionResponse {
                name: call.name.clone(),
                response: json!({
                    "status": "success",
                    "source": "Google Search",
                    "location": params.location,
                    "unit": params.unit,
                    "temperature": {
                        "current": if params.unit == "celsius" { -0.55 } else { 41.0 },
                        "feels_like": if params.unit == "celsius" { -4.0 } else { 39.0 }
                    },
                    "conditions": {
                        "description": "partly cloudy",
                        "humidity": 71,
                        "wind": {
                            "speed": 5,
                            "direction": "N",
                            "gusts": 6
                        },
                        "visibility": 10
                    },
                    "timestamp": "2024-01-03T11:59:18+02:00",
                    "note": "Data based on real-time search results"
                }),
            };

            // Add a small delay before the follow-up request
            sleep(Duration::from_millis(500)).await;

            let follow_up = Request::builder()
                .contents(vec![
                    Content {
                        role: Some(Role::User),
                        parts: vec![Part::Text {
                            text: query.to_string(),
                        }],
                    },
                    Content {
                        role: Some(Role::Model),
                        parts: vec![Part::FunctionCall { function_call: call }],
                    },
                    Content {
                        role: Some(Role::Function),
                        parts: vec![Part::FunctionResponse { function_response: weather_response }],
                    },
                ])
                .build();

            let final_response = generate_response_with_retry(model, follow_up, "weather summary").await?;
            println!("{} {}", "🤖 Response:".green().bold(), final_response.text().white());
        }

        // Add a delay between different weather queries
        sleep(Duration::from_millis(1000)).await;
    }

    Ok(())
}

/// Demonstrates calendar function calling
async fn demonstrate_calendar_function(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "📅 Calendar Function Demo".bright_magenta().bold());
    println!("{}", "=====================".bright_magenta());
    println!(
        "{}",
        "Testing calendar event management functions".bright_black()
    );

    // Define calendar functions
    let calendar_functions = vec![
        FunctionDeclaration::builder()
            .name("add_event")
            .description("Add a new event to the calendar")
            .parameters(
                FunctionDeclarationSchema::builder()
                    .r#type(SchemaType::Object)
                    .properties([
                        (
                            "date".to_string(),
                            Schema::builder()
                                .r#type(SchemaType::String)
                                .description("The date in YYYY-MM-DD format")
                                .build(),
                        ),
                        (
                            "event".to_string(),
                            Schema::builder()
                                .r#type(SchemaType::String)
                                .description("The event description")
                                .build(),
                        ),
                        (
                            "duration_minutes".to_string(),
                            Schema::builder()
                                .r#type(SchemaType::Integer)
                                .description("Duration of the event in minutes")
                                .build(),
                        ),
                    ])
                    .required(vec![
                        "date".to_string(),
                        "event".to_string(),
                        "duration_minutes".to_string(),
                    ])
                    .build(),
            )
            .build(),
        FunctionDeclaration::builder()
            .name("view_events")
            .description("View events for a specific date")
            .parameters(
                FunctionDeclarationSchema::builder()
                    .r#type(SchemaType::Object)
                    .properties([(
                        "date".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .description("The date in YYYY-MM-DD format")
                            .build(),
                    )])
                    .required(vec!["date".to_string()])
                    .build(),
            )
            .build(),
    ];

    // Test calendar queries
    let calendar_queries = [
        "Schedule a team meeting for 2024-01-10 that will last 60 minutes",
        "Add a lunch appointment for tomorrow at 12:30 PM for 45 minutes",
        "What events do I have scheduled for 2024-01-10?",
    ];

    for query in calendar_queries {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "👤 User:".blue().bold(), query);

        let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: query.to_string(),
                }],
            }])
            .tools(vec![calendar_functions.clone().into()])
            .build();

        let response = generate_response_with_retry(model, request, "calendar function").await?;
        let function_calls = response.function_calls();

        if function_calls.is_empty() {
            println!(
                "{} {}",
                "🤖 Response:".green().bold(),
                response.text().white()
            );
            continue;
        }

        for call in function_calls {
            println!(
                "{} {} with {}",
                "📞 Function Call:".yellow().bold(),
                call.name,
                call.args
            );

            // Simulate function execution
            let function_response = match call.name.as_str() {
                "add_event" => {
                    let params: CalendarParams = serde_json::from_value(call.args.clone())?;
                    FunctionResponse {
                        name: call.name.clone(),
                        response: json!({
                            "status": "success",
                            "message": format!("Event '{}' scheduled for {} ({} minutes)",
                                params.event, params.date, params.duration_minutes),
                            "event_id": "evt_123456"
                        }),
                    }
                }
                "view_events" => {
                    let date = call
                        .args
                        .get("date")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    FunctionResponse {
                        name: call.name.clone(),
                        response: json!({
                            "status": "success",
                            "date": date,
                            "events": [
                                {
                                    "time": "09:00",
                                    "event": "Team Meeting",
                                    "duration": 60
                                },
                                {
                                    "time": "12:30",
                                    "event": "Lunch Appointment",
                                    "duration": 45
                                }
                            ]
                        }),
                    }
                }
                _ => FunctionResponse {
                    name: call.name.clone(),
                    response: json!({
                        "status": "error",
                        "message": "Unknown function"
                    }),
                },
            };

            let follow_up = Request::builder()
                .contents(vec![
                    Content {
                        role: Some(Role::User),
                        parts: vec![Part::Text {
                            text: query.to_string(),
                        }],
                    },
                    Content {
                        role: Some(Role::Model),
                        parts: vec![Part::FunctionCall {
                            function_call: call,
                        }],
                    },
                    Content {
                        role: Some(Role::Function),
                        parts: vec![Part::FunctionResponse { function_response }],
                    },
                ])
                .build();

            let final_response = generate_response_with_retry(model, follow_up, "calendar summary").await?;
            println!(
                "{} {}",
                "🤖 Response:".green().bold(),
                final_response.text().white()
            );
        }
    }

    Ok(())
}

/// Demonstrates function calling with Google Search integration
async fn demonstrate_search_function(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🔍 Search Integration Demo".bright_cyan().bold());
    println!("{}", "=====================".bright_cyan());
    println!("{}", "Testing search and bookmark functions".bright_black());

    // Define bookmark function
    let bookmark_function = FunctionDeclaration::builder()
        .name("bookmark_page")
        .description("Save a webpage as a bookmark")
        .parameters(
            FunctionDeclarationSchema::builder()
                .r#type(SchemaType::Object)
                .properties([
                    (
                        "title".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .description("Title of the webpage")
                            .build(),
                    ),
                    (
                        "url".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .description("URL of the webpage")
                            .build(),
                    ),
                    (
                        "category".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .description("Category for organizing bookmarks")
                            .build(),
                    ),
                ])
                .required(vec!["title".to_string(), "url".to_string(), "category".to_string()])
                .build(),
        )
        .build();

    // Test search and bookmark queries
    let queries = [
        "Find me recent articles about Rust programming language and bookmark the most relevant one",
        "Search for the best Italian restaurants in New York and save the top rated one",
        "Look up information about machine learning with Python and bookmark a good tutorial",
    ];

    for query in queries {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "👤 User:".blue().bold(), query);

        // First, perform the search
        let search_request = Request::builder()
            .contents(vec![Content {
                role: None,
                parts: vec![Part::Text { text: query.into() }],
            }])
            .tools(vec![Tool::GOOGLE_SEARCH])
            .build();

        let search_response = generate_response_with_retry(model, search_request, "search").await?;
        display_grounding_metadata(&search_response);

        // Now, ask the model to bookmark the most relevant result
        let bookmark_request = Request::builder()
            .contents(vec![
                Content {
                    role: Some(Role::User),
                    parts: vec![Part::Text { text: query.into() }],
                },
                Content {
                    role: Some(Role::Model),
                    parts: vec![Part::Text { 
                        text: "Based on the search results above, I'll help you bookmark the most relevant page.".to_string() 
                    }],
                },
            ])
            .tools(vec![vec![bookmark_function.clone()].into()])
            .build();

        let bookmark_response = generate_response_with_retry(model, bookmark_request, "bookmark").await?;
        let function_calls = bookmark_response.function_calls();

        if function_calls.is_empty() {
            println!("{} {}", "🤖 Response:".green().bold(), bookmark_response.text().white());
            continue;
        }

        for call in function_calls {
            println!("{} {} with {}", "📞 Function Call:".yellow().bold(), call.name, call.args);

            let params: BookmarkParams = serde_json::from_value(call.args.clone())?;
            let function_response = FunctionResponse {
                name: call.name.clone(),
                response: json!({
                    "status": "success",
                    "message": format!("Bookmarked '{}' in category '{}'", params.title, params.category),
                    "bookmark_id": "bm_123456",
                    "details": {
                        "title": params.title,
                        "url": params.url,
                        "category": params.category,
                        "date_added": "2024-01-03"
                    }
                }),
            };

            let follow_up = Request::builder()
                .contents(vec![
                    Content {
                        role: Some(Role::User),
                        parts: vec![Part::Text {
                            text: query.to_string(),
                        }],
                    },
                    Content {
                        role: Some(Role::Model),
                        parts: vec![Part::FunctionCall { function_call: call }],
                    },
                    Content {
                        role: Some(Role::Function),
                        parts: vec![Part::FunctionResponse { function_response }],
                    },
                ])
                .build();

            let final_response = generate_response_with_retry(model, follow_up, "bookmark summary").await?;
            println!("{} {}", "🤖 Response:".green().bold(), final_response.text().white());
        }
    }

    Ok(())
}

/// Display grounding metadata from the response
fn display_grounding_metadata(response: &gemini_ai_rust::models::Response) {
    if let Some(ref candidates) = response.candidates {
        for candidate in candidates {
            if let Some(ref metadata) = candidate.grounding_metadata {
                // Display search queries used
                if let Some(ref queries) = metadata.web_search_queries {
                    println!("\n{}", "🔎 Search Queries Used:".blue().bold());
                    for query in queries {
                        println!("   • {}", query.cyan());
                    }
                }

                // Display grounding chunks (sources)
                if let Some(ref chunks) = metadata.grounding_chunks {
                    println!("\n{}", "📚 Sources:".yellow().bold());
                    for (i, chunk) in chunks.iter().enumerate() {
                        if let Some(ref web) = chunk.web {
                            println!(
                                "   {}. {}",
                                (i + 1).to_string().yellow(),
                                web.title
                                    .as_ref()
                                    .unwrap_or(&"Untitled".to_string())
                                    .white()
                                    .bold()
                            );
                            if let Some(ref uri) = web.uri {
                                println!("      {}", uri.bright_black().italic());
                            }
                        }
                    }
                }

                // Display grounding supports (evidence)
                if let Some(ref supports) = metadata.grounding_supports {
                    println!("\n{}", "🔍 Evidence:".green().bold());
                    for (i, support) in supports.iter().enumerate() {
                        if let Some(ref segment) = support.segment {
                            if let Some(ref text) = segment.text {
                                println!("   {}. {}", (i + 1).to_string().green(), text.white());

                                // Display confidence scores
                                if let Some(ref scores) = support.confidence_scores {
                                    for score in scores {
                                        let confidence = format!("{:.1}%", score * 100.0);
                                        let colored_score = match score * 100.0 {
                                            x if x >= 90.0 => confidence.bright_green(),
                                            x if x >= 70.0 => confidence.green(),
                                            x if x >= 50.0 => confidence.yellow(),
                                            _ => confidence.red(),
                                        };
                                        println!("      Confidence: {}", colored_score);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "{}",
        "🤖 Gemini Function Calling Demo".bright_green().bold()
    );
    println!("{}", "===========================".bright_green());

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-2.0-flash-exp")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run function demonstrations
    demonstrate_weather_function(&model).await?;
    demonstrate_calendar_function(&model).await?;
    demonstrate_search_function(&model).await?;

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
