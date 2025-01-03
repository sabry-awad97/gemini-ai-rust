use dotenv::dotenv;
use gemini_ai_rust::{
    models::{
        Content, FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration,
        FunctionDeclarationSchema, FunctionResponse, Part, Request, Role, Schema, SchemaType,
        ToolConfig,
    },
    GenerativeModel,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// Define a custom function parameter struct
#[derive(Debug, Serialize, Deserialize)]
struct WeatherParams {
    location: String,
    unit: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-2.0-flash-exp")?;

    // Define a function that the model can call
    let get_weather = FunctionDeclaration::builder()
        .name("get_weather")
        .description("Get the simulated weather data for a location (for demo purposes)")
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
                            .r#enum(vec!["celsius".to_string(), "fahrenheit".to_string()])
                            .description("The temperature unit to use")
                            .build(),
                    ),
                ])
                .required(vec!["location".to_string(), "unit".to_string()])
                .build(),
        )
        .build();

    println!("=== Function Calling Example ===\n");
    println!("Asking about weather in London...\n");

    // Create a request with a user message and the function declaration
    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "Let's test the function calling with simulated weather data. What's the current temperature in London using celsius unit?".into(),
            }],
        }])
        .tools(vec![vec![get_weather].into()])
        .build();

    // Get the initial response
    let response = model.generate_response(request).await?;

    // Get function calls from the response
    let function_calls = response.function_calls();

    if function_calls.is_empty() {
        println!("Model Response: {}", response.text());
        return Ok(());
    }

    // Handle function calls
    for call in function_calls {
        // Parse the function parameters
        let params: WeatherParams = serde_json::from_value(call.args.clone())?;

        println!("Model called function: {}", call.name);
        println!("Arguments: {:?}\n", params);

        // Simulate getting weather data
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

        // Create function response
        let weather_response = FunctionResponse {
            name: call.name.clone(),
            response: weather_data,
        };

        // Create a new request with the function response
        let follow_up = Request::builder()
            .contents(vec![
                Content {
                    role: Some(Role::User),
                    parts: vec![Part::Text {
                        text: "Let's test the function calling with simulated weather data. What's the current temperature in London?".into(),
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

        // Get the final response
        let final_response = model.generate_response(follow_up).await?;
        println!("Final Response: {}", final_response.text());
    }

    println!("\n=== Function Calling with Lights ===\n");

    // Define the available functions
    let functions = vec![
        FunctionDeclaration::builder()
            .name("enable_lights")
            .description("Turn on the lighting system.")
            .parameters(
                FunctionDeclarationSchema::builder()
                    .r#type(SchemaType::Object)
                    .properties([(
                        "confirm".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::Boolean)
                            .description("Confirm turning on the lights".to_string())
                            .build(),
                    )])
                    .build(),
            )
            .build(),
        FunctionDeclaration::builder()
            .name("set_light_color")
            .description("Set the light color. Lights must be enabled for this to work.")
            .parameters(
                FunctionDeclarationSchema::builder()
                    .r#type(SchemaType::Object)
                    .properties([(
                        "rgb_hex".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::String)
                            .description(
                                "The light color as a 6-digit hex string, e.g. ff0000 for red."
                                    .to_string(),
                            )
                            .build(),
                    )])
                    .required(vec!["rgb_hex".to_string()])
                    .build(),
            )
            .build(),
        FunctionDeclaration::builder()
            .name("stop_lights")
            .description("Turn off the lighting system.")
            .parameters(
                FunctionDeclarationSchema::builder()
                    .r#type(SchemaType::Object)
                    .properties([(
                        "confirm".to_string(),
                        Schema::builder()
                            .r#type(SchemaType::Boolean)
                            .description("Confirm turning off the lights".to_string())
                            .build(),
                    )])
                    .build(),
            )
            .build(),
    ];

    // Test different prompts
    let prompts = vec![
        "What can you do?",
        "Turn on the lights and make them red",
        "Change the color to blue",
        "Turn everything off",
    ];

    for prompt in prompts {
        println!("\nUser: {}", prompt);
        println!("Assistant:");

        // Create request with system instruction and functions
        let request = Request::builder()
                .system_instruction(Some("You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks.".into()))
                .contents(vec![
                    Content {
                        role: Some(Role::User),
                        parts: vec![Part::Text {
                            text: prompt.into(),
                        }],
                    },
                ])
                .tools(vec![functions.clone().into()])
                .build();

        // Get response
        let response = model.generate_response(request).await?;

        // Handle function calls if any
        let function_calls = response.function_calls();
        if function_calls.is_empty() {
            println!("{}", response.text());
            continue;
        }

        // Process each function call
        for call in function_calls {
            println!("Called function: {} with args: {}", call.name, call.args);

            // Simulate function execution
            let function_response = match call.name.as_str() {
                "enable_lights" => serde_json::from_value(json!({
                    "name": "enable_lights",
                    "response": {
                        "status": "success",
                        "message": "Lights turned on"
                    }
                }))?,
                "set_light_color" => serde_json::from_value(json!({
                    "name": "set_light_color",
                    "response": {
                        "status": "success",
                        "message": "Light color updated",
                        "color": call.args.get("rgb_hex").unwrap_or(&json!("unknown"))
                    }
                }))?,
                "stop_lights" => serde_json::from_value(json!({
                    "name": "stop_lights",
                    "response": {
                        "status": "success",
                        "message": "Lights turned off"
                    }
                }))?,
                _ => serde_json::from_value(json!({
                    "name": call.name,
                    "response": {
                        "status": "error",
                        "message": "Unknown function"
                    }
                }))?,
            };

            // Create follow-up request with function response
            let follow_up = Request::builder()
                    .system_instruction(Some("You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks.".into()))
                    .contents(vec![
                        Content {
                            role: Some(Role::User),
                            parts: vec![Part::Text {
                                text: prompt.into(),
                            }],
                        },
                        Content {
                            role: Some(Role::Model),
                            parts: vec![Part::FunctionCall { function_call: call }],
                        },
                        Content {
                            role: Some(Role::Function),
                            parts: vec![Part::FunctionResponse {
                                function_response,
                            }],
                        },
                    ])
                    .build();

            // Get final response
            let final_response = model.generate_response(follow_up).await?;
            println!("{}", final_response.text());
        }
    }

    println!("\n=== Function Calling with Function Calling config ===\n");

    // Create request with system instruction and functions
    let request = Request::builder()
        .system_instruction(Some("You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks.".into()))
        .contents(vec![
            Content {
                role: Some(Role::User),
                parts: vec![Part::Text {
                    text: "What can you do?".into(),
                }],
            },
        ])
        .tools(vec![functions.clone().into()])
        .tool_config(ToolConfig::builder()
            .function_calling_config(FunctionCallingConfig::builder()
                .mode(FunctionCallingMode::None)
                .build()
            )
            .build()
        )
        .build();

    let response = model.generate_response(request).await?;

    println!("Response: {}", response.text());

    Ok(())
}
