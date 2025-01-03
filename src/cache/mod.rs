//! Module for managing cached content in the Gemini AI system

use crate::models::{Content, Part, Role};
use reqwest;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The base URL for the cache API
const CACHE_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Error types for cache operations
#[derive(thiserror::Error, Debug)]
pub enum CacheError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    /// Failed to read file from disk
    #[error("Failed to read file: {0}")]
    FileReadError(#[from] std::io::Error),
    /// Invalid MIME type for the file
    #[error("Invalid MIME type: {0}")]
    MimeTypeError(String),
    /// Generic cache operation error
    #[error("Cache operation failed: {0}")]
    OperationError(String),
}

/// Information about a cached content
#[derive(Debug, Deserialize, Serialize)]
pub struct CacheInfo {
    /// The resource name of the cached content
    pub name: String,
    /// The cached content
    pub contents: Vec<Content>,
    /// Optional system instruction for the cached content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Time-to-live duration in seconds
    pub ttl: String,
    /// Creation time of the cached content
    pub create_time: Option<String>,
    /// Last update time of the cached content
    pub update_time: Option<String>,
    /// Expiration time of the cached content
    pub expire_time: Option<String>,
}

/// Request to create cached content
#[derive(Debug, Serialize)]
pub struct CreateCacheRequest {
    /// The model to use for the cached content
    pub model: String,
    /// The content to cache
    pub contents: Vec<Content>,
    /// Optional system instruction for the cached content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Time-to-live duration in seconds
    pub ttl: String,
}

/// Manager for cache operations
pub struct CacheManager {
    /// The HTTP client used for cache operations
    client: reqwest::Client,
    /// The API key used for authentication
    api_key: String,
}

impl CacheManager {
    /// Creates a new instance of the cache manager
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }

    /// Creates a new cached content from a file
    ///
    /// # Arguments
    ///
    /// * `model`: The model to use for the cached content
    /// * `file_path`: The path to the file to cache
    /// * `system_instruction`: Optional system instruction for the cached content
    /// * `ttl`: Time-to-live duration in seconds
    pub async fn create_cache_from_file(
        &self,
        model: impl Into<String>,
        file_path: impl AsRef<Path>,
        system_instruction: Option<Content>,
        ttl: impl Into<String>,
    ) -> Result<CacheInfo, CacheError> {
        let file_path = file_path.as_ref();

        // Create cache request
        let request = CreateCacheRequest {
            model: model.into(),
            contents: vec![Content {
                parts: vec![Part::image_from_path(file_path)?],
                role: Some(Role::User),
            }],
            system_instruction,
            ttl: ttl.into(),
        };

        // Send request
        let url = format!("{}/cachedContents", CACHE_API_URL);
        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.api_key)])
            .json(&request)
            .send()
            .await?;

        // Check if response is an error
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(CacheError::OperationError(format!(
                "Request failed with status {}: {}",
                status, error_text
            )));
        }

        // Parse response
        let cache_info = response.json().await?;
        Ok(cache_info)
    }

    /// Lists all cached contents
    pub async fn list_caches(&self) -> Result<Vec<CacheInfo>, CacheError> {
        let url = format!("{}/cachedContents", CACHE_API_URL);
        let response = self
            .client
            .get(&url)
            .query(&[("key", &self.api_key)])
            .send()
            .await?;

        // Check if response is an error
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(CacheError::OperationError(format!(
                "Request failed with status {}: {}",
                status, error_text
            )));
        }

        #[derive(Deserialize)]
        struct ListResponse {
            cached_contents: Vec<CacheInfo>,
        }

        let list = response.json::<ListResponse>().await?;
        Ok(list.cached_contents)
    }

    /// Gets information about a specific cached content
    ///
    /// # Arguments
    ///
    /// * `name`: The resource name of the cached content
    pub async fn get_cache(&self, name: &str) -> Result<CacheInfo, CacheError> {
        let url = format!("{}/{}", CACHE_API_URL, name);
        let response = self
            .client
            .get(&url)
            .query(&[("key", &self.api_key)])
            .send()
            .await?;

        // Check if response is an error
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(CacheError::OperationError(format!(
                "Request failed with status {}: {}",
                status, error_text
            )));
        }

        let cache_info = response.json().await?;
        Ok(cache_info)
    }

    /// Updates the TTL of a cached content
    ///
    /// # Arguments
    ///
    /// * `name`: The resource name of the cached content
    /// * `ttl`: Time-to-live duration in seconds
    pub async fn update_cache_ttl(
        &self,
        name: &str,
        ttl: impl Into<String>,
    ) -> Result<CacheInfo, CacheError> {
        let url = format!("{}/{}", CACHE_API_URL, name);
        let response = self
            .client
            .patch(&url)
            .query(&[("key", &self.api_key)])
            .json(&serde_json::json!({ "ttl": ttl.into() }))
            .send()
            .await?;

        // Check if response is an error
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(CacheError::OperationError(format!(
                "Request failed with status {}: {}",
                status, error_text
            )));
        }

        let cache_info = response.json().await?;
        Ok(cache_info)
    }

    /// Deletes a cached content
    ///
    /// # Arguments
    ///
    /// * `name`: The resource name of the cached content
    pub async fn delete_cache(&self, name: &str) -> Result<(), CacheError> {
        let url = format!("{}/{}", CACHE_API_URL, name);
        let response = self
            .client
            .delete(&url)
            .query(&[("key", &self.api_key)])
            .send()
            .await?;

        // Check if response is an error
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(CacheError::OperationError(format!(
                "Request failed with status {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }
}
