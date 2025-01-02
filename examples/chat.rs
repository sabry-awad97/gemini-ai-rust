use futures::StreamExt;
use gemini_ai_rust::{chat::ChatSession, GenerativeModel};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();

    // Create a client from environment variables
    let model = GenerativeModel::from_env("gemini-1.5-flash")?;

    println!("=== Regular Chat Example ===\n");

    // Create a chat session with a system instruction
    let mut chat = ChatSession::new(model.clone()).with_system_instruction(
        "You are a helpful AI assistant with expertise in Rust programming. \
        You provide clear, concise answers with code examples when relevant.",
    );

    // Have a conversation
    let messages = [
        "Hello! Can you help me understand Rust's ownership system?",
        "Can you show me a simple example?",
        "How does borrowing work in this context?",
    ];

    for message in messages {
        println!("\nUser: {}", message);
        print!("Assistant: ");
        let response = chat.send_message(message).await?;
        println!("{}", response);
    }

    println!("\n=== Streaming Chat Example ===\n");

    // Create a new chat session for streaming
    let mut streaming_chat = ChatSession::new(model).with_system_instruction(
        "You are a creative storyteller who crafts engaging short stories.",
    );

    let prompt =
        "Tell me a short story about a programmer who discovers they can talk to their computer.";
    println!("User: {}", prompt);
    print!("Assistant: ");

    // Start streaming chat
    let mut stream = streaming_chat.send_message_streaming(prompt).await?;

    // Print the streaming responses
    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => {
                print!("{}", response.text());
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    println!("\n");

    Ok(())
}
