use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{
        Content, FunctionDeclaration, FunctionDeclarationSchema, FunctionResponse, Part, Request,
        Role, Schema, SchemaType,
    },
    GenerativeModel,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// Define custom function parameter structs
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

/// Demonstrates weather function calling
async fn demonstrate_weather_function(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🌤️  Weather Function Demo".bright_blue().bold());
    println!("{}", "=====================".bright_blue());
    println!(
        "{}",
        "Testing weather data retrieval function".bright_black()
    );

    // Define the weather function
    let get_weather = FunctionDeclaration::builder()
        .name("get_weather")
        .description("Get the simulated weather data for a location")
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

        let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: query.to_string(),
                }],
            }])
            .tools(vec![vec![get_weather.clone()].into()])
            .build();

        let response = model.generate_response(request).await?;
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

            let params: WeatherParams = serde_json::from_value(call.args.clone())?;

            // Simulate weather data
            let weather_data = json!({
                "location": params.location,
                "temperature": 18,
                "unit": params.unit,
                "condition": "partly cloudy",
                "humidity": 65,
                "wind_speed": 12,
                "wind_direction": "NE",
                "note": "This is simulated data for demonstration purposes"
            });

            let weather_response = FunctionResponse {
                name: call.name.clone(),
                response: weather_data,
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
                        parts: vec![Part::FunctionResponse {
                            function_response: weather_response,
                        }],
                    },
                ])
                .build();

            let final_response = model.generate_response(follow_up).await?;
            println!(
                "{} {}",
                "🤖 Response:".green().bold(),
                final_response.text().white()
            );
        }
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

        let response = model.generate_response(request).await?;
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

            let final_response = model.generate_response(follow_up).await?;
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

    // Define search and bookmark functions
    let functions = vec![
        FunctionDeclaration::builder()
            .name("search_web")
            .description("Search the web for information")
            .parameters(
                FunctionDeclarationSchema::builder()
                    .r#type(SchemaType::Object)
                    .properties([
                        (
                            "query".to_string(),
                            Schema::builder()
                                .r#type(SchemaType::String)
                                .description("The search query")
                                .build(),
                        ),
                        (
                            "max_results".to_string(),
                            Schema::builder()
                                .r#type(SchemaType::Integer)
                                .description("Maximum number of results to return (1-5)")
                                .build(),
                        ),
                    ])
                    .required(vec!["query".to_string(), "max_results".to_string()])
                    .build(),
            )
            .build(),
        FunctionDeclaration::builder()
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
                    .required(vec![
                        "title".to_string(),
                        "url".to_string(),
                        "category".to_string(),
                    ])
                    .build(),
            )
            .build(),
    ];

    // Test search and bookmark queries
    let queries = [
        "Find me recent articles about Rust programming language and bookmark the most relevant one",
        "Search for the best Italian restaurants in New York and save the top rated one",
        "Look up information about machine learning with Python and bookmark a good tutorial",
    ];

    for query in queries {
        println!("\n{}", "━".repeat(50).bright_black());
        println!("{} {}", "👤 User:".blue().bold(), query);

        let request = Request::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: query.to_string(),
                }],
            }])
            .tools(vec![functions.clone().into()])
            .build();

        let response = model.generate_response(request).await?;
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
                "search_web" => {
                    let params: SearchResultParams = serde_json::from_value(call.args.clone())?;
                    FunctionResponse {
                        name: call.name.clone(),
                        response: json!({
                            "status": "success",
                            "query": params.query,
                            "results": [
                                {
                                    "title": "Getting Started with Rust: A Beginner's Guide",
                                    "url": "https://example.com/rust-guide",
                                    "snippet": "A comprehensive guide to learning Rust programming language...",
                                    "date": "2024-01-02"
                                },
                                {
                                    "title": "Best Practices for Rust Development",
                                    "url": "https://example.com/rust-best-practices",
                                    "snippet": "Learn about memory safety, ownership, and other Rust concepts...",
                                    "date": "2024-01-01"
                                }
                            ]
                        }),
                    }
                }
                "bookmark_page" => {
                    let params: BookmarkParams = serde_json::from_value(call.args.clone())?;
                    FunctionResponse {
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

            let final_response = model.generate_response(follow_up).await?;
            println!(
                "{} {}",
                "🤖 Response:".green().bold(),
                final_response.text().white()
            );
        }
    }

    Ok(())
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
