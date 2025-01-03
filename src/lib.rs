#![deny(missing_docs)]

//! A Rust client library for the Google Gemini AI API.
//!
//! This library provides a simple and idiomatic way to interact with Google's Gemini AI API.
//! It handles authentication, request construction, and response parsing.
//!
//! # Example
//! ```no_run
//! use gemini_ai_rust::client::GenerativeModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = GenerativeModel::from_env("gemini-1.5-flash")?;
//!     let response = client.generate_content("Explain how AI works").await?;
//!     response.display();
//!     Ok(())
//! }
//! ```

pub mod cache;
pub mod chat;
pub mod client;
pub mod error;
pub mod file;
pub mod models;

pub use client::GenerativeModel;
pub use file::GoogleAIFileManager;
