//! Client implementation for the Gemini AI API.

use futures::StreamExt;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::models::{
    BatchEmbedContentRequest, BatchEmbedContentResponse, EmbedContentRequest, EmbedContentResponse,
    ListModelsResponse, ModelInfo, ResponseStream,
};
use crate::{
    error::GoogleGenerativeAIError,
    models::{ModelParams, Request, RequestType, Response, TokenCountResponse},
};

/// Default API endpoint for Google's Generative AI service
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// Default API version
const DEFAULT_API_VERSION: &str = "v1beta";
/// Default channel buffer size for streaming responses
const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 16;
/// Default buffer capacity for JSON parsing
const DEFAULT_JSON_BUFFER_CAPACITY: usize = 4096;

/// A client for interacting with the Gemini AI API.
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    api_key: String,
    params: ModelParams,
    client: reqwest::Client,
}

impl GenerativeModel {
    /// Creates a new GenerativeModel with the specified API key and model.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for authentication
    /// * `params` - The model parameters
    pub fn new(api_key: impl Into<String>, params: impl Into<ModelParams>) -> Self {
        Self {
            api_key: api_key.into(),
            params: params.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new GenerativeModel from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `GOOGLE_API_KEY` - The API key for authentication
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gemini-1.5-flash")
    ///
    /// # Errors
    ///
    /// Returns an error if the required environment variable is not set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, GoogleGenerativeAIError> {
        let api_key = std::env::var("GOOGLE_API_KEY")?;
        Ok(Self::new(
            api_key,
            ModelParams::builder().model(model).build(),
        ))
    }

    /// Makes a request to the Gemini AI API.
    ///
    /// # Arguments
    ///
    /// * `request` - The request to send to the API
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    async fn make_request<T>(
        &self,
        url: &str,
        request: T,
    ) -> Result<reqwest::Response, GoogleGenerativeAIError>
    where
        T: Serialize,
    {
        let response = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(GoogleGenerativeAIError::new(format!(
                "Request failed with status {}: {}",
                status, error_body
            )));
        }

        Ok(response)
    }

    /// Sends the HTTP request and processes the response.
    async fn send_request<T: serde::de::DeserializeOwned, R>(
        &self,
        url: &str,
        request: R,
    ) -> Result<T, GoogleGenerativeAIError>
    where
        R: Serialize,
    {
        Ok(self.make_request(url, request).await?.json::<T>().await?)
    }

    fn build_url(&self, model: &str, request_type: RequestType) -> String {
        format!(
            "{}/{}/models/{}:{}?key={}",
            DEFAULT_BASE_URL, DEFAULT_API_VERSION, model, request_type, self.api_key
        )
    }

    /// Generates content using the Gemini AI API.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to generate content from
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn send_message(
        &self,
        prompt: impl Into<String>,
    ) -> Result<Response, GoogleGenerativeAIError> {
        let mut request = Request::with_prompt(prompt);
        request.generation_config = request
            .generation_config
            .or_else(|| self.params.generation_config.clone());
        let url = self.build_url(self.params.model.as_str(), RequestType::GenerateContent);

        self.send_request(&url, request).await
    }

    /// Generates response using the Gemini AI API with a system instruction.
    ///
    /// # Arguments
    ///
    /// * `system_instruction` - The system instruction for the model
    /// * `prompt` - The text prompt to generate content from
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn generate_response(
        &self,
        request: impl Into<Request>,
    ) -> Result<Response, GoogleGenerativeAIError> {
        let url = self.build_url(self.params.model.as_str(), RequestType::GenerateContent);
        let mut request = request.into();
        request.generation_config = request
            .generation_config
            .or_else(|| self.params.generation_config.clone());
        self.send_request(&url, request).await
    }

    /// Generates streaming content using the Gemini AI API.
    pub async fn stream_generate_response(
        &self,
        request: impl Into<Request>,
    ) -> Result<ResponseStream, GoogleGenerativeAIError> {
        let url = self.build_url(
            self.params.model.as_str(),
            RequestType::StreamGenerateContent,
        );
        let response = self.make_request(&url, request.into()).await?;

        let (tx, rx) = mpsc::channel(DEFAULT_CHANNEL_BUFFER_SIZE);
        let mut stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::with_capacity(DEFAULT_JSON_BUFFER_CAPACITY);
            let mut in_object = false;
            let mut object_depth = 0;
            let mut in_string = false;
            let mut escaped = false;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => match std::str::from_utf8(&chunk) {
                        Ok(chunk_str) => {
                            for c in chunk_str.chars() {
                                match c {
                                    '"' if !escaped => {
                                        in_string = !in_string;
                                        buffer.push(c);
                                    }
                                    '\\' if !escaped => {
                                        escaped = true;
                                        buffer.push(c);
                                    }
                                    '{' if !in_string => {
                                        if !in_object {
                                            in_object = true;
                                            buffer.clear();
                                        }
                                        object_depth += 1;
                                        buffer.push(c);
                                    }
                                    '}' if !in_string => {
                                        object_depth -= 1;
                                        buffer.push(c);

                                        if object_depth == 0 && in_object {
                                            in_object = false;
                                            match serde_json::from_str(&buffer) {
                                                Ok(response) => {
                                                    if tx.send(Ok(response)).await.is_err() {
                                                        return;
                                                    }
                                                }
                                                Err(e) => {
                                                    if tx
                                                        .send(Err(GoogleGenerativeAIError::new(
                                                            format!(
                                                                "Failed to parse response: {}",
                                                                e
                                                            ),
                                                        )))
                                                        .await
                                                        .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                            }
                                            buffer.clear();
                                            buffer.reserve(DEFAULT_JSON_BUFFER_CAPACITY);
                                        }
                                    }
                                    '[' if !in_string && !in_object => buffer.clear(),
                                    ']' if !in_string && !in_object => buffer.clear(),
                                    _ => {
                                        if in_object {
                                            buffer.push(c);
                                        }
                                        escaped = false;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            if let Err(e) = tx
                                .send(Err(GoogleGenerativeAIError::new(format!(
                                    "UTF-8 decode error: {}",
                                    e
                                ))))
                                .await
                            {
                                eprintln!("Error sending error: {}", e);
                            }
                        }
                    },
                    Err(e) => {
                        if let Err(e) = tx
                            .send(Err(GoogleGenerativeAIError::new(e.to_string())))
                            .await
                        {
                            eprintln!("Error sending error: {}", e);
                        }
                    }
                }
            }
        });

        Ok(ResponseStream::new(rx))
    }

    /// Counts the number of tokens in the given content.
    ///
    /// # Arguments
    ///
    /// * `request` - The request containing the content to count tokens for
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn count_tokens(
        &self,
        request: impl Into<Request>,
    ) -> Result<TokenCountResponse, GoogleGenerativeAIError> {
        let url = self.build_url(self.params.model.as_str(), RequestType::CountTokens);
        let mut request = request.into();
        request.generation_config = request
            .generation_config
            .or_else(|| self.params.generation_config.clone());
        self.send_request(&url, request).await
    }

    /// List all available models
    pub async fn list_models(&self) -> Result<ListModelsResponse, GoogleGenerativeAIError> {
        let url = format!("{}/{}/models", DEFAULT_BASE_URL, DEFAULT_API_VERSION);
        let url = format!("{}?key={}", url, self.api_key);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(GoogleGenerativeAIError::new(format!(
                "Failed to list models: {} - {}",
                status, error_body
            )));
        }

        Ok(response.json().await?)
    }

    /// Get information about a specific model
    pub async fn get_model_info(
        &self,
        model_name: &str,
    ) -> Result<ModelInfo, GoogleGenerativeAIError> {
        let url = format!(
            "{}/{}/models/{}",
            DEFAULT_BASE_URL, DEFAULT_API_VERSION, model_name
        );
        let url = format!("{}?key={}", url, self.api_key);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(GoogleGenerativeAIError::new(format!(
                "Failed to get model {}: {} - {}",
                model_name, status, error_body
            )));
        }

        Ok(response.json().await?)
    }

    /// Generate embeddings for the given content using the specified model
    ///
    /// # Arguments
    /// * `request` - The content and optional parameters for the embedding request
    ///
    /// # Returns
    /// A Result containing either the embedding response or a GoogleGenerativeAIError
    pub async fn embed_content(
        &self,
        model: &str,
        request: impl Into<EmbedContentRequest>,
    ) -> Result<EmbedContentResponse, GoogleGenerativeAIError> {
        let url = self.build_url(model, RequestType::EmbedContent);
        self.send_request(&url, request.into()).await
    }

    /// Batch embed multiple contents in a single request
    ///
    /// # Arguments
    /// * `model` - The model to use for embedding
    /// * `requests` - A vector of embedding requests to process in batch
    ///
    /// # Returns
    /// A result containing the batch embedding response or an error
    pub async fn batch_embed_contents(
        &self,
        model: &str,
        requests: Vec<EmbedContentRequest>,
    ) -> Result<BatchEmbedContentResponse, GoogleGenerativeAIError> {
        let url = self.build_url(model, RequestType::BatchEmbedContents);
        let request = BatchEmbedContentRequest { requests };
        self.send_request(&url, request).await
    }
}
