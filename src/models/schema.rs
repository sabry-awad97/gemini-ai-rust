use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

/// The type of a property in a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaType {
    /// String type.
    String,
    /// Number type.
    Number,
    /// Integer type.
    Integer,
    /// Boolean type.
    Boolean,
    /// Array type.
    Array,
    /// Object type.
    Object,
}

/// A schema for a function parameter.
///
/// This struct represents the JSON Schema format used to define parameters for function declarations.
/// It supports various types, formats, descriptions, and nested schemas for complex types.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(doc)]
pub struct Schema {
    /// Optional. The type of the property.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub r#type: Option<SchemaType>,

    /// Optional. The format of the property.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub format: Option<String>,

    /// Optional. The description of the property.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub description: Option<String>,

    /// Optional. Whether the property is nullable.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub nullable: Option<bool>,

    /// Optional. The items of the property.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub items: Option<Box<Schema>>, // Use Box for recursive struct

    /// Optional. The enum of the property.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub r#enum: Option<Vec<String>>, // 'enum' is a reserved keyword in Rust

    /// Optional. Map of Schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub properties: Option<std::collections::HashMap<String, Schema>>, // Use HashMap for property map

    /// Optional. The required properties.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub required: Option<Vec<String>>,

    /// Optional. The example of the property.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub example: Option<serde_json::Value>, // Use serde_json::Value for unknown types
}
