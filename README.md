# Gemini AI Rust Client

[![Crates.io](https://img.shields.io/crates/v/gemini-ai-rust)](https://crates.io/crates/gemini-ai-rust)
[![Documentation](https://docs.rs/gemini-ai-rust/badge.svg)](https://docs.rs/gemini-ai-rust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, async Rust client library for Google's Gemini AI API. This library provides an idiomatic Rust interface for interacting with Gemini's powerful language models, supporting both simple text generation and advanced features like streaming responses, function calling, and file operations.

## Features

- 🚀 **Async-first Design**: Built on Tokio for efficient async operations
- 🔄 **Streaming Support**: Real-time streaming responses with proper backpressure handling
- 🛠️ **Function Calling**: Advanced function declaration and handling capabilities
- 📁 **File Operations**: Comprehensive file management with progress tracking
- 🔒 **Safety Settings**: Configurable model safety parameters
- 💾 **Caching**: Built-in response caching mechanisms
- 🎯 **Rate Limiting**: Automatic retry with exponential backoff
- 🎨 **Type Safety**: Strong type system with proper error handling

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
gemini-ai-rust = "0.1.0"
```

## Quick Start

```rust
use gemini_ai_rust::GenerativeModel;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize client from environment variable
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;
    
    // Generate content
    let response = client.generate_content("Explain how AI works").await?;
    println!("{}", response.text());
    
    Ok(())
}
```

## Environment Setup

Set your Google API key in your environment:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

Or use a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Advanced Usage Examples

### Streaming Responses

```rust
use gemini_ai_rust::GenerativeModel;

async fn stream_example() -> Result<(), Box<dyn Error>> {
    let client = GenerativeModel::from_env("gemini-1.5-flash")?;
    let mut stream = client.stream_generate_response("Tell me a story").await?;
    
    while let Some(response) = stream.next().await {
        print!("{}", response?.text());
    }
    Ok(())
}
```

## Features in Detail

### File Operations
- Upload and process files with progress tracking
- Support for various file formats
- Efficient streaming of large files

### Safety Settings
- Configurable content filtering
- Customizable safety thresholds
- Comprehensive safety categories

### Caching
- Response caching for improved performance
- Configurable cache duration
- Memory and disk caching options

### Model Configuration
- Customizable model parameters
- Temperature and top-k/top-p sampling
- Stop sequence configuration

## Examples

The [examples](examples/) directory contains comprehensive examples demonstrating various features:

- Basic text generation
- Streaming responses
- Chat functionality
- Function calling
- File operations
- Safety settings
- Caching mechanisms
- Model configuration
- Google Search integration
- Code execution

## Documentation

For detailed documentation and API reference, visit [docs.rs/gemini-ai-rust](https://docs.rs/gemini-ai-rust).

## Error Handling

The library uses the `thiserror` crate for comprehensive error handling. All errors are properly typed and propagated through the `Result` type.

```rust
use gemini_ai_rust::error::GoogleGenerativeAIError;

match client.generate_content("prompt").await {
    Ok(response) => println!("Success: {}", response.text()),
    Err(GoogleGenerativeAIError::RateLimitExceeded) => {
        println!("Rate limit exceeded, retrying...");
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini AI team for their excellent API
- The Rust community for their invaluable crates and support

## Support

If you encounter any issues or have questions, please:
1. Check the [documentation](https://docs.rs/gemini-ai-rust)
2. Look through [existing issues](https://github.com/sabry-awad97/gemini-ai-rust/issues)
3. Open a new issue if needed

---

Built with ❤️ using Rust
