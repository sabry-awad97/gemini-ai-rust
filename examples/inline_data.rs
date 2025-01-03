use colored::*;
use dotenv::dotenv;
use gemini_ai_rust::{
    models::{Content, GenerationConfig, Part, Request, ResponseSchema, Role, SchemaType},
    GenerativeModel,
};
use serde::{Deserialize, Serialize};
use std::error::Error;

const IMAGE_PATH: &str = "examples/inline_data.jpg";

#[derive(Debug, Serialize, Deserialize)]
struct InventoryItem {
    serial: Option<String>,
    batch_no: Option<String>,
    expiry: Option<String>,
    product: String,
    quantity: i32,
    price: f64,
    discount: Option<f64>,
    total: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PharmacyData {
    items: Vec<InventoryItem>,
    current_balance: f64,
    previous_balance: f64,
    total_due: f64,
    account_number: i32,
    print_date: String,
    print_time: String,
    last_payment_date: String,
    last_payment_value: f64,
    cashier: String,
    user: String,
}

/// Display inventory items in a formatted table
fn display_inventory(items: &[InventoryItem]) {
    println!("\n{}", "📦 Inventory Items".bright_blue().bold());
    println!("{}", "═".repeat(100).bright_blue());
    println!(
        "{:<10} {:<15} {:<12} {:<30} {:<8} {:<10} {:<10}",
        "Serial#".yellow(),
        "Batch#".yellow(),
        "Expiry".yellow(),
        "Product".yellow(),
        "Qty".yellow(),
        "Price".yellow(),
        "Total".yellow()
    );
    println!("{}", "─".repeat(100).bright_black());

    for item in items {
        println!(
            "{:<10} {:<15} {:<12} {:<30} {:<8} {:<10.2} {:<10}",
            item.serial.as_deref().unwrap_or("-").bright_cyan(),
            item.batch_no.as_deref().unwrap_or("-").bright_cyan(),
            item.expiry.as_deref().unwrap_or("N/A").bright_cyan(),
            if item.product.starts_with("**") {
                item.product[2..].bright_red()
            } else {
                item.product.white()
            },
            item.quantity.to_string().bright_green(),
            item.price,
            item.total.bright_yellow()
        );
    }
    println!("{}", "═".repeat(100).bright_blue());
}

/// Display financial summary
fn display_financial_summary(data: &PharmacyData) {
    println!("\n{}", "💰 Financial Summary".bright_green().bold());
    println!("{}", "═".repeat(50).bright_green());
    println!(
        "{:<20} {:>10.2} {}",
        "Current Balance:".bright_white(),
        data.current_balance,
        "EGP".bright_black()
    );
    println!(
        "{:<20} {:>10.2} {}",
        "Previous Balance:".bright_white(),
        data.previous_balance,
        "EGP".bright_black()
    );
    println!(
        "{:<20} {:>10.2} {}",
        "Total Due:".bright_white(),
        data.total_due,
        "EGP".bright_black()
    );
    println!("{}", "─".repeat(50).bright_black());
    println!(
        "{:<20} {:>10}",
        "Account Number:".bright_white(),
        data.account_number.to_string().bright_cyan()
    );
}

/// Display transaction details
fn display_transaction_details(data: &PharmacyData) {
    println!("\n{}", "🕒 Transaction Details".bright_magenta().bold());
    println!("{}", "═".repeat(50).bright_magenta());
    println!(
        "{:<20} {}",
        "Print Date:".bright_white(),
        data.print_date.bright_cyan()
    );
    println!(
        "{:<20} {}",
        "Print Time:".bright_white(),
        data.print_time.bright_cyan()
    );
    println!(
        "{:<20} {} ({} EGP)",
        "Last Payment:".bright_white(),
        data.last_payment_date.bright_cyan(),
        data.last_payment_value.to_string().bright_yellow()
    );
    println!("{}", "─".repeat(50).bright_black());
    println!(
        "{:<20} {}",
        "Cashier:".bright_white(),
        data.cashier.bright_green()
    );
    println!(
        "{:<20} {}",
        "User:".bright_white(),
        data.user.bright_green()
    );
}

/// Demonstrates inventory data extraction from image
async fn demonstrate_inventory_extraction(model: &GenerativeModel) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "🏥 Pharmacy Inventory Demo".bright_blue().bold());
    println!("{}", "========================".bright_blue());
    println!(
        "{}",
        "Extracting structured inventory data from image".bright_black()
    );

    // Create schema for inventory items
    let inventory_schema = ResponseSchema::builder()
        .r#type(SchemaType::Object)
        .properties([
            (
                "items".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::Array)
                    .items(
                        ResponseSchema::builder()
                            .r#type(SchemaType::Object)
                            .properties([
                                (
                                    "serial".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::String)
                                        .description("Serial number of the item")
                                        .build(),
                                ),
                                (
                                    "batch_no".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::String)
                                        .description("Batch number of the item")
                                        .build(),
                                ),
                                (
                                    "expiry".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::String)
                                        .description("Expiry date of the item")
                                        .build(),
                                ),
                                (
                                    "product".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::String)
                                        .description("Name of the product")
                                        .build(),
                                ),
                                (
                                    "quantity".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::Integer)
                                        .description("Quantity in stock")
                                        .build(),
                                ),
                                (
                                    "price".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::Number)
                                        .description("Unit price")
                                        .build(),
                                ),
                                (
                                    "total".to_string(),
                                    ResponseSchema::builder()
                                        .r#type(SchemaType::String)
                                        .description("Total value")
                                        .build(),
                                ),
                            ])
                            .required(vec![
                                "product".into(),
                                "quantity".into(),
                                "price".into(),
                                "total".into(),
                            ])
                            .build(),
                    )
                    .build(),
            ),
            (
                "current_balance".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::Number)
                    .description("Current account balance")
                    .build(),
            ),
            (
                "previous_balance".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::Number)
                    .description("Previous account balance")
                    .build(),
            ),
            (
                "total_due".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::Number)
                    .description("Total amount due")
                    .build(),
            ),
            (
                "account_number".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::Integer)
                    .description("Account identifier")
                    .build(),
            ),
            (
                "print_date".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::String)
                    .description("Date of printing")
                    .build(),
            ),
            (
                "print_time".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::String)
                    .description("Time of printing")
                    .build(),
            ),
            (
                "last_payment_date".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::String)
                    .description("Date of last payment")
                    .build(),
            ),
            (
                "last_payment_value".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::Number)
                    .description("Amount of last payment")
                    .build(),
            ),
            (
                "cashier".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::String)
                    .description("Name of the cashier")
                    .build(),
            ),
            (
                "user".to_string(),
                ResponseSchema::builder()
                    .r#type(SchemaType::String)
                    .description("Name of the user")
                    .build(),
            ),
        ])
        .required(vec![
            "items".into(),
            "current_balance".into(),
            "previous_balance".into(),
            "total_due".into(),
            "account_number".into(),
            "print_date".into(),
            "print_time".into(),
            "cashier".into(),
            "user".into(),
        ])
        .build();

    // Create request with image and schema
    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![
                Part::text(
                    "Extract the inventory data from this image and convert it to JSON format",
                ),
                Part::image_from_path(IMAGE_PATH)?,
            ],
        }])
        .generation_config(
            GenerationConfig::builder()
                .response_mime_type("application/json")
                .response_schema(inventory_schema)
                .build(),
        )
        .build();

    // Generate and display response
    println!("\n{}", "🔄 Processing Image...".yellow().bold());
    let response = model.generate_response(request).await?;

    // Parse and display the data
    let data: PharmacyData = serde_json::from_str(&response.text())?;
    display_inventory(&data.items);
    display_financial_summary(&data);
    display_transaction_details(&data);

    println!(
        "\n{}",
        "✨ Data extraction completed!".bright_green().bold()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "🤖 Gemini Vision Demo".bright_green().bold());
    println!("{}", "==================".bright_green());

    // Load environment variables
    dotenv().ok();
    println!("{}", "✓ Environment loaded".green());

    // Create client from environment variables
    let model = GenerativeModel::from_env("gemini-2.0-flash-exp")?;
    println!("{}", "✓ Gemini model initialized".green());

    // Demonstrate inventory extraction
    demonstrate_inventory_extraction(&model).await?;

    Ok(())
}
