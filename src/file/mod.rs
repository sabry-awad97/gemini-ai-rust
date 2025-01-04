//! File models for the Gemini AI API.

use mime_guess;
use reqwest;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;
use tokio;

const FILE_API_VERSION: &str = "v1beta";
const FILE_API_URL: &str = "https://generativelanguage.googleapis.com";

/// Represents possible errors that can occur during file operations.
#[derive(Error, Debug)]
pub enum FileError {
    /// Error occurred during HTTP request.
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    /// Failed to get file size from the filesystem.
    #[error("Failed to get file size: {0}")]
    FileSizeError(io::Error),
    /// Failed to read file contents.
    #[error("Failed to read file: {0}")]
    FileReadError(io::Error),
    /// Failed to determine MIME type for the file.
    #[error("Invalid MIME type: {0}")]
    MimeTypeError(String),
    /// Error occurred during file upload process.
    #[error("Upload failed: {0}")]
    UploadError(String),
    /// Invalid file ID provided.
    #[error("Invalid file ID: {0}")]
    InvalidFileId(String),
    /// Error occurred during file processing.
    #[error("File processing error: {0}")]
    ProcessingError(String),
}

/// Information about a file stored in the Gemini AI system.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileInfo {
    /// Unique identifier for the file.
    pub name: String,
    /// Optional display name for the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// MIME type of the file content.
    pub mime_type: String,
    /// Size of the file in bytes as a string.
    pub size_bytes: String,
    /// Time when the file was created.
    pub create_time: String,
    /// Time when the file was last updated.
    pub update_time: String,
    /// Optional expiration time for the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_time: Option<String>,
    /// Optional SHA256 hash of the file content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_hash: Option<String>,
    /// URI that can be used to reference this file in API calls.
    pub uri: String,
    /// Current processing state of the file.
    pub state: FileState,
    /// Optional error message if the file processing failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Optional metadata for video files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_metadata: Option<serde_json::Value>,
    /// Optional description of the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Represents the processing state of a file in the system.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FileState {
    /// State is not specified.
    #[serde(rename = "STATE_UNSPECIFIED")]
    Unspecified,
    /// File is currently being processed.
    Processing,
    /// File is processed and ready for use.
    Active,
    /// An error occurred during file processing.
    Failed,
}

impl std::fmt::Display for FileState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileState::Unspecified => write!(f, "Unspecified"),
            FileState::Processing => write!(f, "Processing"),
            FileState::Active => write!(f, "Active"),
            FileState::Failed => write!(f, "Failed"),
        }
    }
}

/// Manager for handling file operations with the Gemini AI API.
#[derive(Debug)]
pub struct GoogleAIFileManager {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GoogleAIFileManager {
    /// Creates a new instance of the Google AI File Manager.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The Google AI API key to use for authentication. Can be any type that can be converted into a String.
    ///
    /// # Returns
    ///
    /// Returns a new `GoogleAIFileManager` instance configured with the provided API key.
    ///
    /// # Example
    ///
    /// ```
    /// use gemini_ai_rust::file::GoogleAIFileManager;
    ///
    /// let api_key = "your-api-key-here";
    /// let file_manager = GoogleAIFileManager::new(api_key);
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        let base_url =
            std::env::var("GOOGLE_BASE_URL").unwrap_or_else(|_| FILE_API_URL.to_string());

        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url,
        }
    }

    /// Creates a new instance of the file manager using the GOOGLE_API_KEY environment variable.
    pub fn from_env() -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .expect("GOOGLE_API_KEY environment variable must be set must be set");
        Self::new(api_key)
    }

    /// Deletes all files with the specified display name.
    /// Returns the number of files deleted.
    pub async fn delete_files_by_display_name(
        &self,
        display_name: &str,
    ) -> Result<usize, FileError> {
        let files = self.list_files().await?;
        let mut deleted_count = 0;

        for file in files {
            if let Some(name) = &file.display_name {
                if name == display_name {
                    self.delete_file(&file.name).await?;
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }

    /// Uploads a file to the Gemini AI system.
    ///
    /// # Arguments
    /// * `file_path` - Path to the file to upload
    /// * `display_name` - Optional display name for the file
    ///
    /// # Returns
    /// Information about the uploaded file.
    pub async fn upload_file<A: AsRef<Path>, I: Into<Option<String>>>(
        &self,
        file_path: A,
        display_name: I,
    ) -> Result<FileInfo, FileError> {
        let file_path = file_path.as_ref();
        let file_size = fs::metadata(file_path)
            .map_err(FileError::FileSizeError)?
            .len();

        let mime_type = mime_guess::from_path(file_path)
            .first()
            .ok_or_else(|| {
                FileError::MimeTypeError(format!("Unknown MIME type for {:?}", file_path))
            })?
            .to_string();

        // Initial resumable upload request
        let upload_url = format!("{}/upload/{}/files", self.base_url, FILE_API_VERSION);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("X-Goog-Upload-Protocol", "resumable".parse().unwrap());
        headers.insert("X-Goog-Upload-Command", "start".parse().unwrap());
        headers.insert(
            "X-Goog-Upload-Header-Content-Length",
            file_size.to_string().parse().unwrap(),
        );
        headers.insert(
            "X-Goog-Upload-Header-Content-Type",
            mime_type.parse().unwrap(),
        );

        let metadata = serde_json::json!({
            "file": {
                "display_name": display_name.into().unwrap_or_else(|| file_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unnamed")
                    .to_string())
            }
        });

        let response = self
            .client
            .post(&upload_url)
            .query(&[("key", &self.api_key)])
            .headers(headers)
            .json(&metadata)
            .send()
            .await?;

        let upload_url = response
            .headers()
            .get("x-goog-upload-url")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| FileError::UploadError("Missing upload URL".into()))?
            .to_string();

        // Upload file content
        let file_content = tokio::fs::read(file_path)
            .await
            .map_err(FileError::FileReadError)?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Length", file_size.to_string().parse().unwrap());
        headers.insert("X-Goog-Upload-Offset", "0".parse().unwrap());
        headers.insert("X-Goog-Upload-Command", "upload, finalize".parse().unwrap());

        let response = self
            .client
            .post(&upload_url)
            .headers(headers)
            .body(file_content)
            .send()
            .await?;

        let response_text = response.text().await?;
        println!("Response: {}", response_text);

        #[derive(Deserialize)]
        struct FileResponse {
            file: FileInfo,
        }

        let file_response: FileResponse = serde_json::from_str(&response_text).map_err(|e| {
            FileError::UploadError(format!(
                "Failed to parse response: {}. Response: {}",
                e, response_text
            ))
        })?;
        Ok(file_response.file)
    }

    /// Retrieves information about a file by its name.
    pub async fn get_file(&self, name: &str) -> Result<FileInfo, FileError> {
        let url = format!("{}/{}/files/{}", self.base_url, FILE_API_VERSION, name);
        let response = self
            .client
            .get(&url)
            .query(&[("key", &self.api_key)])
            .send()
            .await?;

        let file_info: FileInfo = response.json().await?;
        Ok(file_info)
    }

    /// Deletes a file from the system.
    pub async fn delete_file(&self, file_id: &str) -> Result<(), FileError> {
        let url = format!(
            "{}/{}/files/{}?key={}",
            self.base_url,
            FILE_API_VERSION,
            parse_file_id(file_id)?,
            self.api_key
        );
        self.client
            .delete(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;

        Ok(())
    }

    /// Lists all files available in the system.
    pub async fn list_files(&self) -> Result<Vec<FileInfo>, FileError> {
        let url = format!("{}/{}/files", self.base_url, FILE_API_VERSION);
        let response = self
            .client
            .get(&url)
            .query(&[("key", &self.api_key)])
            .send()
            .await?;

        #[derive(Deserialize)]
        struct ListResponse {
            files: Vec<FileInfo>,
        }

        let list_response: ListResponse = response.json().await?;
        Ok(list_response.files)
    }

    /// Waits for a file to finish processing, with configurable retries and delay.
    ///
    /// # Arguments
    /// * `name` - Name of the file to wait for
    /// * `max_retries` - Maximum number of times to check the file state
    /// * `delay_ms` - Delay in milliseconds between retries
    ///
    /// # Returns
    /// The file information once processing is complete or an error if processing fails.
    pub async fn wait_for_file_processing(
        &self,
        name: &str,
        max_retries: u32,
        delay_ms: u64,
    ) -> Result<FileInfo, FileError> {
        for _ in 0..max_retries {
            let file_info = self.get_file(name).await?;
            match file_info.state {
                FileState::Active => return Ok(file_info),
                FileState::Failed => {
                    return Err(FileError::ProcessingError(format!(
                        "File {} processing failed",
                        name
                    )))
                }
                FileState::Processing | FileState::Unspecified => {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    continue;
                }
            }
        }
        Err(FileError::ProcessingError(format!(
            "Timeout waiting for file {} to process",
            name
        )))
    }
}

fn parse_file_id(file_id: &str) -> Result<&str, FileError> {
    if let Some(stripped) = file_id.strip_prefix("files/") {
        Ok(stripped)
    } else if !file_id.is_empty() {
        Ok(file_id)
    } else {
        Err(FileError::InvalidFileId(
            "File ID must not be empty".to_string(),
        ))
    }
}
