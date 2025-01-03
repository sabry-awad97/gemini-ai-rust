use dotenv::dotenv;
use gemini_ai_rust::{
    cache::CacheManager,
    models::{Content, Part, Request, Role},
    GenerativeModel,
};
use std::path::PathBuf;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Get API key from environment
    let api_key =
        std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable must be set");

    // Create managers
    let cache_manager = CacheManager::new(&api_key);
    let model = GenerativeModel::from_env("gemini-1.5-flash-001")?;

    // Example file path (you should create this file with some text content)
    let example_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("test.txt");

    println!("Creating cache from file: {:?}", example_file);

    // Create cache with system instruction
    let system_instruction = Some(Content {
        role: Some(Role::System),
        parts: vec![Part::text("You are an expert at analyzing transcripts.")],
    });

    let cache_info = cache_manager
        .create_cache_from_file(
            "models/gemini-1.5-flash-001",
            &example_file,
            system_instruction,
            "300s",
        )
        .await?;

    println!("Cache created: {}", cache_info.name);

    // Generate content using the cached content
    println!("\nGenerating content using cache...");
    let request = Request::builder()
        .contents(vec![Content {
            role: Some(Role::User),
            parts: vec![Part::text("Please summarize this transcript")],
        }])
        .cached_content(cache_info.name.clone())
        .build();

    let mut stream = model.stream_generate_response(request).await?;
    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => print!("{}", response.text()),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    println!();

    // List all caches
    println!("\nListing all caches:");
    let caches = cache_manager.list_caches().await?;
    for cache in &caches {
        println!("- {} (expires: {:?})", cache.name, cache.expire_time);
    }

    // Get specific cache
    println!("\nGetting cache info:");
    let cache = cache_manager.get_cache(&cache_info.name).await?;
    println!("Cache TTL: {}", cache.ttl);

    // Update cache TTL
    println!("\nUpdating cache TTL to 600s...");
    let updated_cache = cache_manager
        .update_cache_ttl(&cache_info.name, "600s")
        .await?;
    println!("New TTL: {}", updated_cache.ttl);

    // Delete cache
    println!("\nDeleting cache...");
    cache_manager.delete_cache(&cache_info.name).await?;
    println!("Cache deleted successfully!");

    Ok(())
}
