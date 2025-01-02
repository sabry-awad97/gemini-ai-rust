//! Function declarations and related types for the Gemini AI API.

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::schema::{Schema, SchemaType};

/// A function declaration schema that can be passed to the model.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclarationSchema {
    /// The type of the parameter.
    pub r#type: SchemaType,

    /// The properties of the parameter.
    #[builder(setter(into))]
    pub properties: std::collections::HashMap<String, Schema>,

    /// Optional. Description of the parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub description: Option<String>,

    /// Optional. Array of required parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub required: Option<Vec<String>>,
}

/// A function declaration that can be passed to the model.
///
/// The model may decide to call a subset of these functions by populating
/// [`FunctionCall`] in the response. The user should provide a [`FunctionResponse`]
/// for each function call in the next turn. Based on the function responses,
/// the model will generate the final response back to the user.
///
/// Maximum 64 function declarations can be provided.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    /// The name of the function.
    #[builder(setter(into))]
    pub name: String,

    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub description: Option<String>,

    /// The parameters of the function in JSON Schema format.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub parameters: Option<FunctionDeclarationSchema>,
}

/// A function call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to pass to the function.
    pub args: serde_json::Value,
}

/// A response to a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    /// The name of the function that was called.
    pub name: String,

    /// The response from the function.
    pub response: serde_json::Value,
}
