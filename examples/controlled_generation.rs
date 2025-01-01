use dotenv::dotenv;
use gemini_ai_rust::{
    client::GenerativeModel,
    models::{Content, GenerationConfig, Part, Request, ResponseSchema, SchemaType},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Create a new client from environment variables
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;

    // Prepare the request
    let request = Request::builder()
        .contents(vec![Content {
            parts: vec![Part::Text {
                text: "List 5 popular cookie recipes".into(),
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
                                            .build(),
                                    ),
                                    (
                                        "description".to_string(),
                                        ResponseSchema::builder()
                                            .r#type(SchemaType::String)
                                            .build(),
                                    ),
                                ])
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .build();

    let response = client.generate_content(request).await?;

    // Display the response
    println!("{}", response.text());

    let response = client
        .send_message(
            r#"List a few popular cookie recipes using this JSON schema:

            Recipe = {\"recipe_name\": str, \"description\": str}
            Return: list[Recipe]"#,
        )
        .await?;

    println!("{}", "*".repeat(10));

    println!("{}", response.text());

    Ok(())
}
