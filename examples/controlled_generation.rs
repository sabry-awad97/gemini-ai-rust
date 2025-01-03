use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, GenerationConfig, Part, Request, ResponseSchema, Role, SchemaType},
    GenerativeModel,
};
use std::error::Error;

/// Demonstrates recipe schema generation
async fn demonstrate_recipe_schema(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🍪 Recipe Schema Demo".bright_blue().bold());
    println!("{}", "==================".bright_blue());
    println!(
        "{}",
        "Generating structured recipe data in JSON format".bright_black()
    );

    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "List 3 popular cookie recipes with detailed descriptions".into(),
            }],
        }])
        .generation_config(
            GenerationConfig::builder()
                .response_mime_type("application/json")
                .response_schema(
                    ResponseSchema::builder()
                        .r#type(SchemaType::Array)
                        .items(
                            ResponseSchema::builder()
                                .r#type(SchemaType::Object)
                                .properties([
                                    (
                                        "recipe_name".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .description("Name of the cookie recipe")
                                            .build(),
                                    ),
                                    (
                                        "description".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .description("Brief description of the cookie")
                                            .build(),
                                    ),
                                    (
                                        "difficulty".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .enum_values(vec![
                                                "Easy".into(),
                                                "Medium".into(),
                                                "Hard".into(),
                                            ])
                                            .build(),
                                    ),
                                ])
                                .required(vec![
                                    "recipe_name".into(),
                                    "description".into(),
                                    "difficulty".into(),
                                ])
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .build();

    println!("\n{}", "📤 Generated Response:".green().bold());
    let response = model.generate_response(request).await?;
    println!("{}", response.text().white());

    Ok(())
}

/// Demonstrates book recommendation schema
async fn demonstrate_book_schema(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!(
        "\n{}",
        "📚 Book Recommendations Schema Demo"
            .bright_magenta()
            .bold()
    );
    println!("{}", "==============================".bright_magenta());
    println!(
        "{}",
        "Generating structured book recommendations in JSON format".bright_black()
    );

    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "Recommend 3 science fiction books for beginners".into(),
            }],
        }])
        .generation_config(
            GenerationConfig::builder()
                .response_mime_type("application/json")
                .response_schema(
                    ResponseSchema::builder()
                        .r#type(SchemaType::Array)
                        .items(
                            ResponseSchema::builder()
                                .r#type(SchemaType::Object)
                                .properties([
                                    (
                                        "title".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .description("Book title")
                                            .build(),
                                    ),
                                    (
                                        "author".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .description("Book author")
                                            .build(),
                                    ),
                                    (
                                        "year_published".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::Integer)
                                            .description("Year of publication")
                                            .build(),
                                    ),
                                    (
                                        "themes".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::Array)
                                            .items(
                                                ResponseSchema::builder()
                                                    .r#type(SchemaType::String)
                                                    .build(),
                                            )
                                            .build(),
                                    ),
                                ])
                                .required(vec![
                                    "title".into(),
                                    "author".into(),
                                    "year_published".into(),
                                    "themes".into(),
                                ])
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .build();

    println!("\n{}", "📤 Generated Response:".green().bold());
    let response = model.generate_response(request).await?;
    println!("{}", response.text().white());

    Ok(())
}

/// Demonstrates weather forecast schema
async fn demonstrate_weather_schema(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!(
        "\n{}",
        "🌤️  Weather Forecast Schema Demo".bright_cyan().bold()
    );
    println!("{}", "===========================".bright_cyan());
    println!(
        "{}",
        "Generating structured weather forecast in JSON format".bright_black()
    );

    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: "Generate a 3-day weather forecast for New York".into(),
            }],
        }])
        .generation_config(
            GenerationConfig::builder()
                .response_mime_type("application/json")
                .response_schema(
                    ResponseSchema::builder()
                        .r#type(SchemaType::Array)
                        .items(
                            ResponseSchema::builder()
                                .r#type(SchemaType::Object)
                                .properties([
                                    (
                                        "day".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::Integer)
                                            .build(),
                                    ),
                                    (
                                        "condition".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .enum_values(vec![
                                                "Sunny".into(),
                                                "Cloudy".into(),
                                                "Rainy".into(),
                                                "Snowy".into(),
                                            ])
                                            .build(),
                                    ),
                                    (
                                        "temperature".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::Object)
                                            .properties([
                                                (
                                                    "high".to_string(),
                                                    ResponseSchema::builder()
                                                        .r#type(SchemaType::Integer)
                                                        .build(),
                                                ),
                                                (
                                                    "low".to_string(),
                                                    ResponseSchema::builder()
                                                        .r#type(SchemaType::Integer)
                                                        .build(),
                                                ),
                                            ])
                                            .build(),
                                    ),
                                ])
                                .required(vec![
                                    "day".into(),
                                    "condition".into(),
                                    "temperature".into(),
                                ])
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .build();

    println!("\n{}", "📤 Generated Response:".green().bold());
    let response = model.generate_response(request).await?;
    println!("{}", response.text().white());

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "{}",
        "🤖 Gemini Controlled Generation Demo".bright_green().bold()
    );
    println!("{}", "================================".bright_green());

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Run schema demonstrations
    demonstrate_recipe_schema(&model).await?;
    demonstrate_book_schema(&model).await?;
    demonstrate_weather_schema(&model).await?;

    println!("\n{}", "✨ Demo completed successfully!".green().bold());
    Ok(())
}
