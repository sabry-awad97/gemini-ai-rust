#![deny(missing_docs)]

//! A Rust client library for the Google Gemini AI API.
//!
//! This library provides a simple and idiomatic way to interact with Google's Gemini AI API.
//! It handles authentication, request construction, and response parsing.

pub mod cache;
pub mod chat;
pub mod client;
pub mod error;
pub mod file;
pub mod models;

pub use client::GenerativeModel;
pub use file::GoogleAIFileManager;
