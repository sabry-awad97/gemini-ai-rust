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
    #[builder(default, setter(strip_option, into))]
    pub name: Option<String>,

    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub description: Option<String>,

    /// The parameters of the function in JSON Schema format.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub parameters: Option<FunctionDeclarationSchema>,
}

impl Default for FunctionDeclaration {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionDeclaration {
    /// Creates a new function declaration builder.
    pub fn new() -> Self {
        FunctionDeclaration::builder().build()
    }

    /// Sets the name of the function.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the description of the function.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Parses a schema type and any modifiers from a type string
    /// Format: "type[:modifier(value1,value2,...)]"
    fn parse_schema_type_with_modifiers(type_str: &str) -> (SchemaType, Option<Vec<String>>) {
        println!("\nParsing type string: {}", type_str);
        let parts: Vec<&str> = type_str.split(':').collect();
        let base_type = Self::parse_schema_type(parts[0]);
        println!("Base type: {:?}", base_type);

        // Check for enum modifier
        if let Some(modifier) = parts.get(1) {
            println!("Found modifier: {}", modifier);
            if modifier.starts_with("enum(") && modifier.ends_with(')') {
                let mut enum_str = modifier.trim_start_matches("enum(");
                // Only trim the last closing parenthesis
                if enum_str.ends_with(')') {
                    enum_str = &enum_str[..enum_str.len() - 1];
                }
                println!("Enum string: '{}'", enum_str);

                let enum_values = if enum_str.is_empty() {
                    vec![]
                } else {
                    enum_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect::<Vec<_>>()
                };
                println!("Enum values: {:?}", enum_values);

                (base_type, Some(enum_values))
            } else {
                println!("Modifier is not an enum");
                (base_type, None)
            }
        } else {
            println!("No modifier found");
            (base_type, None)
        }
    }

    /// Parses a schema type from a string.
    fn parse_schema_type(type_str: &str) -> SchemaType {
        match type_str.trim().to_lowercase().as_str() {
            "string" => SchemaType::String,
            "number" => SchemaType::Number,
            "integer" => SchemaType::Integer,
            "boolean" => SchemaType::Boolean,
            "array" => SchemaType::Array,
            "object" => SchemaType::Object,
            _ => SchemaType::String, // Default to string for unknown types
        }
    }

    /// Parses a parameter definition into a Schema.
    fn parse_parameter(param_str: &str) -> Option<(String, Schema)> {
        let parts: Vec<&str> = param_str.split('|').map(str::trim).collect();

        // First split by comma, but handle the case where we have enum values
        let mut remaining = parts[0];
        let mut base_parts = Vec::new();

        // Extract name (everything up to first comma)
        if let Some((name, rest)) = remaining.split_once(',') {
            base_parts.push(name.trim());
            remaining = rest.trim();

            // Extract type and enum values if present
            if remaining.contains("enum(") {
                if let Some(end_paren) = remaining.rfind(')') {
                    let type_and_enum = &remaining[..=end_paren];
                    base_parts.push(type_and_enum.trim());
                    remaining = &remaining[end_paren + 1..];

                    // Remove leading comma from description if present
                    if remaining.starts_with(',') {
                        remaining = remaining[1..].trim();
                    }
                }
            } else if let Some((type_str, rest)) = remaining.split_once(',') {
                base_parts.push(type_str.trim());
                remaining = rest.trim();
            }

            // Whatever is left is the description
            if !remaining.is_empty() {
                base_parts.push(remaining);
            }
        }

        if base_parts.len() < 2 {
            return None;
        }

        let name = base_parts[0].to_string();
        let type_str = base_parts[1];
        let description = base_parts
            .get(2)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let (schema_type, enum_values) = Self::parse_schema_type_with_modifiers(type_str);

        // Handle object type with properties
        if schema_type == SchemaType::Object && parts.len() > 1 {
            let properties = Self::parse_object_properties(parts[1]);
            let required: Vec<String> = properties.keys().cloned().collect();
            let schema = Schema::builder()
                .r#type(schema_type)
                .description(description)
                .properties(properties)
                .required(required)
                .build();
            Some((name, schema))
        } else {
            let schema = if let Some(values) = enum_values {
                Schema::builder()
                    .r#type(schema_type)
                    .description(description)
                    .enum_values(values)
                    .build()
            } else {
                Schema::builder()
                    .r#type(schema_type)
                    .description(description)
                    .build()
            };
            Some((name, schema))
        }
    }

    /// Parses object properties string into a HashMap of property schemas.
    ///
    /// Format: "prop1:type[:desc], prop2:type[:desc], prop3:{subprop1:type, subprop2:type}"
    fn parse_object_properties(props_str: &str) -> std::collections::HashMap<String, Schema> {
        println!("\nParsing object properties: {}", props_str);
        let mut properties = std::collections::HashMap::new();
        let mut current_prop = String::new();
        let mut brace_count = 0;
        let mut paren_count = 0;

        // First, properly split the properties handling nested braces and parentheses
        let mut props = Vec::new();
        for c in props_str.chars() {
            match c {
                '{' => {
                    brace_count += 1;
                    current_prop.push(c);
                }
                '}' => {
                    brace_count -= 1;
                    current_prop.push(c);
                }
                '(' => {
                    paren_count += 1;
                    current_prop.push(c);
                }
                ')' => {
                    paren_count -= 1;
                    current_prop.push(c);
                }
                ',' if brace_count == 0 && paren_count == 0 => {
                    if !current_prop.trim().is_empty() {
                        println!("Found property: {}", current_prop.trim());
                        props.push(current_prop.trim().to_string());
                        current_prop.clear();
                    }
                }
                _ => {
                    current_prop.push(c);
                }
            }
        }
        if !current_prop.trim().is_empty() {
            println!("Found property: {}", current_prop.trim());
            props.push(current_prop.trim().to_string());
        }

        // Now process each property
        for prop in props {
            let prop = prop.trim();
            println!("\nProcessing property: {}", prop);

            // Check if this is a nested object
            if prop.contains('{') {
                let nested_parts: Vec<&str> = prop.splitn(2, ':').collect();
                if nested_parts.len() == 2 {
                    let prop_name = nested_parts[0].to_string();
                    let mut nested_props_str = nested_parts[1].to_string();
                    println!(
                        "Found nested object - name: {}, props: {}",
                        prop_name, nested_props_str
                    );

                    // Remove outer braces and any trailing comma
                    nested_props_str = nested_props_str
                        .trim_start_matches('{')
                        .trim_end_matches('}')
                        .trim_end_matches(',')
                        .to_string();
                    println!("Cleaned nested props: {}", nested_props_str);

                    let nested_properties = Self::parse_object_properties(&nested_props_str);
                    let required: Vec<String> = nested_properties.keys().cloned().collect();
                    let schema = Schema::builder()
                        .r#type(SchemaType::Object)
                        .properties(nested_properties)
                        .required(required)
                        .build();
                    properties.insert(prop_name, schema);
                }
            } else {
                // Handle basic property
                let mut parts: Vec<&str> = prop.split(':').map(str::trim).collect();
                println!("Basic property parts: {:?}", parts);

                // Clean up any trailing characters from the last part
                if let Some(last) = parts.last_mut() {
                    *last = last.trim_end_matches(['}', ',']);
                }

                if parts.len() >= 2 {
                    let prop_name = parts[0].to_string();

                    // Find where the type string ends and description begins
                    let mut type_end = 1;
                    let mut desc_start = parts.len();

                    for i in 2..parts.len() {
                        if !parts[i].starts_with("enum(") {
                            type_end = i - 1;
                            desc_start = i;
                            break;
                        }
                    }

                    // Join all type parts including any enum modifiers
                    let type_str = parts[1..=type_end].join(":");
                    let description = parts
                        .get(desc_start)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();
                    println!(
                        "Name: {}, Type: {}, Description: {}",
                        prop_name, type_str, description
                    );

                    let (schema_type, enum_values) =
                        Self::parse_schema_type_with_modifiers(&type_str);
                    let schema = Schema::builder()
                        .r#type(schema_type)
                        .description(description)
                        .enum_values(enum_values.unwrap_or_default())
                        .build();
                    properties.insert(prop_name, schema);
                }
            }
        }

        properties
    }

    /// Sets the parameters of the function using a slice of parameter definitions.
    /// Each parameter can be either a basic type or an object with properties.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gemini_ai_rust::models::FunctionDeclaration;
    ///
    /// let func = FunctionDeclaration::new()
    ///     .with_parameters(&[
    ///         // Basic parameters
    ///         "name, string, User's name",
    ///         "age, integer, User's age",
    ///         
    ///         // Object with properties
    ///         "address, object, User's address | street:string:Street name, city:string, country:string",
    ///         
    ///         // Object with nested properties
    ///         "settings, object, User settings | preferences:{theme:string:UI theme, notifications:boolean:Enable notifications}"
    ///     ]);
    /// ```
    pub fn with_parameters(mut self, parameters: &[&str]) -> Self {
        let mut properties = std::collections::HashMap::new();
        let mut required = Vec::new();

        for param_str in parameters {
            if let Some((name, schema)) = Self::parse_parameter(param_str) {
                properties.insert(name.clone(), schema);
                required.push(name);
            }
        }

        self.parameters = Some(
            FunctionDeclarationSchema::builder()
                .r#type(SchemaType::Object)
                .properties(properties)
                .required(required)
                .build(),
        );

        self
    }
}

/// A function call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
pub struct FunctionCall<T = serde_json::Value> {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to pass to the function.
    pub args: T,
}

/// A response to a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
pub struct FunctionResponse<T = serde_json::Value> {
    /// The name of the function that was called.
    pub name: String,

    /// The response from the function.
    pub response: T,
}

/// Specifies how the model should handle function calling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FunctionCallingMode {
    /// The model decides to predict either a function call or a natural language response.
    Auto,
    /// The model is constrained to always predict a function call.
    /// If allowed_function_names is not provided, the model picks from all available function declarations.
    /// If allowed_function_names is provided, the model picks from the set of allowed functions.
    Any,
    /// The model won't predict a function call.
    /// In this case, the model behavior is the same as if you don't pass any function declarations.
    None,
}

/// Configuration for how the model should handle function calling.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// The mode of function calling to use
    pub mode: FunctionCallingMode,
    /// Optional list of allowed function names. Only used when mode is Any.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub allowed_function_names: Option<Vec<String>>,
}

/// A list of function declarations to be used in a chat session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclarationTool {
    /// The list of function declarations
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_declaration_builder() {
        let func = FunctionDeclaration::new()
            .with_name("test_function")
            .with_description("A test function");

        assert_eq!(func.name, Some("test_function".to_string()));
        assert_eq!(func.description, Some("A test function".to_string()));
        assert!(func.parameters.is_none());
    }

    #[test]
    fn test_parameter_parsing_basic() {
        let func = FunctionDeclaration::new().with_parameters(&["location, string, The city name"]);

        let params = func.parameters.unwrap();
        assert_eq!(params.r#type, SchemaType::Object);

        let properties = params.properties;
        assert_eq!(properties.len(), 1);

        let location_param = properties.get("location").unwrap();
        assert_eq!(location_param.r#type, Some(SchemaType::String));
        assert_eq!(
            location_param.description,
            Some("The city name".to_string())
        );
    }

    #[test]
    fn test_parameter_parsing_multiple() {
        let func = FunctionDeclaration::new().with_parameters(&[
            "location, string, The city name",
            "temperature, number, Temperature in degrees",
            "is_metric, boolean, Use metric units",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;
        assert_eq!(properties.len(), 3);

        // Check location parameter
        let location = properties.get("location").unwrap();
        assert_eq!(location.r#type, Some(SchemaType::String));
        assert_eq!(location.description, Some("The city name".to_string()));

        // Check temperature parameter
        let temperature = properties.get("temperature").unwrap();
        assert_eq!(temperature.r#type, Some(SchemaType::Number));
        assert_eq!(
            temperature.description,
            Some("Temperature in degrees".to_string())
        );

        // Check is_metric parameter
        let is_metric = properties.get("is_metric").unwrap();
        assert_eq!(is_metric.r#type, Some(SchemaType::Boolean));
        assert_eq!(is_metric.description, Some("Use metric units".to_string()));

        // Check required parameters
        let required = params.required.unwrap();
        assert_eq!(required.len(), 3);
        assert!(required.contains(&"location".to_string()));
        assert!(required.contains(&"temperature".to_string()));
        assert!(required.contains(&"is_metric".to_string()));
    }

    #[test]
    fn test_parameter_without_description() {
        let func = FunctionDeclaration::new().with_parameters(&["age, integer"]);

        let params = func.parameters.unwrap();
        let properties = params.properties;
        let age_param = properties.get("age").unwrap();

        assert_eq!(age_param.r#type, Some(SchemaType::Integer));
        assert_eq!(age_param.description, Some("".to_string()));
    }

    #[test]
    fn test_parameter_invalid_type() {
        let func = FunctionDeclaration::new().with_parameters(&["data, invalid_type, Some data"]);

        let params = func.parameters.unwrap();
        let properties = params.properties;
        let data_param = properties.get("data").unwrap();

        // Should default to string type for invalid types
        assert_eq!(data_param.r#type, Some(SchemaType::String));
        assert_eq!(data_param.description, Some("Some data".to_string()));
    }

    #[test]
    fn test_parameter_all_types() {
        let func = FunctionDeclaration::new().with_parameters(&[
            "text, string, A string value",
            "count, integer, An integer value",
            "amount, number, A number value",
            "flag, boolean, A boolean value",
            "list, array, An array value",
            "data, object, An object value",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        let type_checks = vec![
            ("text", SchemaType::String),
            ("count", SchemaType::Integer),
            ("amount", SchemaType::Number),
            ("flag", SchemaType::Boolean),
            ("list", SchemaType::Array),
            ("data", SchemaType::Object),
        ];

        for (name, expected_type) in type_checks {
            let param = properties.get(name).unwrap();
            assert_eq!(param.r#type, Some(expected_type));
            assert!(param.description.is_some());
            assert!(!param.description.as_ref().unwrap().is_empty());
        }
    }

    #[test]
    fn test_empty_parameters() {
        let func = FunctionDeclaration::new().with_parameters(&[]);

        let params = func.parameters.unwrap();
        assert_eq!(params.properties.len(), 0);
        assert_eq!(params.required.unwrap().len(), 0);
    }

    #[test]
    fn test_chaining_methods() {
        let func = FunctionDeclaration::new()
            .with_name("weather")
            .with_description("Get weather info")
            .with_parameters(&["city, string, City name"])
            .with_description("Updated description"); // Should override previous description

        assert_eq!(func.name, Some("weather".to_string()));
        assert_eq!(func.description, Some("Updated description".to_string()));
        assert!(func.parameters.is_some());
    }

    #[test]
    fn test_malformed_parameter() {
        let func = FunctionDeclaration::new().with_parameters(&["malformed_param"]); // Missing type and description

        let params = func.parameters.unwrap();
        assert_eq!(params.properties.len(), 0); // Should skip malformed parameter
    }

    #[test]
    fn test_object_with_properties() {
        let func = FunctionDeclaration::new()
            .with_parameters(&[
                "address, object, User's address | street:string:Street name, city:string:City name, country:string"
            ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;
        let address = properties.get("address").unwrap();

        // Check address schema
        assert_eq!(address.r#type, Some(SchemaType::Object));
        assert_eq!(address.description, Some("User's address".to_string()));

        // Check address properties
        let addr_props = address.properties.as_ref().unwrap();

        // Check street property
        let street = addr_props.get("street").unwrap();
        assert_eq!(street.r#type, Some(SchemaType::String));
        assert_eq!(street.description, Some("Street name".to_string()));

        // Check city property
        let city = addr_props.get("city").unwrap();
        assert_eq!(city.r#type, Some(SchemaType::String));
        assert_eq!(city.description, Some("City name".to_string()));

        // Check country property
        let country = addr_props.get("country").unwrap();
        assert_eq!(country.r#type, Some(SchemaType::String));
        assert_eq!(country.description, Some("".to_string()));

        // Check required fields
        let required = address.required.as_ref().unwrap();
        assert!(required.contains(&"street".to_string()));
        assert!(required.contains(&"city".to_string()));
        assert!(required.contains(&"country".to_string()));
    }

    #[test]
    fn test_nested_object_properties() {
        let func = FunctionDeclaration::new()
            .with_parameters(&[
                "settings, object, User preferences | \
                 config:{debug:boolean:Debug mode, level:integer:Log level}, \
                 theme:{colors:{primary:string:UI theme, secondary:string}, mode:string:Theme mode}"
            ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;
        let settings = properties.get("settings").unwrap();

        // Check settings schema
        assert_eq!(settings.r#type, Some(SchemaType::Object));

        let settings_props = settings.properties.as_ref().unwrap();
        let theme = settings_props.get("theme").unwrap();
        assert_eq!(theme.r#type, Some(SchemaType::Object));

        let theme_props = theme.properties.as_ref().unwrap();
        let mode = theme_props.get("mode").unwrap();
        assert_eq!(mode.r#type, Some(SchemaType::String));
        assert_eq!(mode.description, Some("Theme mode".to_string()));

        let colors = theme_props.get("colors").unwrap();
        assert_eq!(colors.r#type, Some(SchemaType::Object));

        let colors_props = colors.properties.as_ref().unwrap();
        let primary = colors_props.get("primary").unwrap();
        assert_eq!(primary.r#type, Some(SchemaType::String));
        assert_eq!(primary.description, Some("UI theme".to_string()));

        let secondary = colors_props.get("secondary").unwrap();
        assert_eq!(secondary.r#type, Some(SchemaType::String));
        assert_eq!(secondary.description, Some("".to_string()));
    }

    #[test]
    fn test_mixed_parameters() {
        let func = FunctionDeclaration::new().with_parameters(&[
            "name, string, User's name",
            "age, integer, User's age",
            "preferences, object, User preferences | theme:string:UI theme, notifications:boolean",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        // Check basic parameters
        let name = properties.get("name").unwrap();
        assert_eq!(name.r#type, Some(SchemaType::String));
        assert_eq!(name.description, Some("User's name".to_string()));

        let age = properties.get("age").unwrap();
        assert_eq!(age.r#type, Some(SchemaType::Integer));
        assert_eq!(age.description, Some("User's age".to_string()));

        // Check object parameter
        let preferences = properties.get("preferences").unwrap();
        assert_eq!(preferences.r#type, Some(SchemaType::Object));
        let pref_props = preferences.properties.as_ref().unwrap();

        let theme = pref_props.get("theme").unwrap();
        assert_eq!(theme.r#type, Some(SchemaType::String));
        assert_eq!(theme.description, Some("UI theme".to_string()));

        let notifications = pref_props.get("notifications").unwrap();
        assert_eq!(notifications.r#type, Some(SchemaType::Boolean));
        assert_eq!(notifications.description, Some("".to_string()));
    }

    #[test]
    fn test_parameter_with_enum() {
        let func = FunctionDeclaration::new().with_parameters(&[
            "mode, string:enum(light,dark), Display mode",
            "unit, string:enum(celsius,fahrenheit), Temperature unit",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        // Check mode parameter
        let mode = properties.get("mode").unwrap();
        assert_eq!(mode.r#type, Some(SchemaType::String));
        assert_eq!(mode.description, Some("Display mode".to_string()));
        assert_eq!(
            mode.enum_values,
            Some(vec!["light".to_string(), "dark".to_string()])
        );

        // Check unit parameter
        let unit = properties.get("unit").unwrap();
        assert_eq!(unit.r#type, Some(SchemaType::String));
        assert_eq!(unit.description, Some("Temperature unit".to_string()));
        assert_eq!(
            unit.enum_values,
            Some(vec!["celsius".to_string(), "fahrenheit".to_string()])
        );
    }

    #[test]
    fn test_enum_with_spaces() {
        let func = FunctionDeclaration::new().with_parameters(&[
            "theme, string:enum(light theme, dark theme), Display theme",
            "size, string:enum(extra small,  small  ,medium,  large ), Size option",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        let theme = properties.get("theme").unwrap();
        assert_eq!(theme.r#type, Some(SchemaType::String));
        assert_eq!(theme.description, Some("Display theme".to_string()));
        assert_eq!(
            theme.enum_values,
            Some(vec!["light theme".to_string(), "dark theme".to_string()])
        );

        let size = properties.get("size").unwrap();
        assert_eq!(size.r#type, Some(SchemaType::String));
        assert_eq!(size.description, Some("Size option".to_string()));
        assert_eq!(
            size.enum_values,
            Some(vec![
                "extra small".to_string(),
                "small".to_string(),
                "medium".to_string(),
                "large".to_string()
            ])
        );
    }

    #[test]
    fn test_nested_object_with_enum() {
        let func =
            FunctionDeclaration::new().with_parameters(&["settings, object, User settings | \
             theme:{mode:string:enum(light,dark):Theme mode, \
                   colors:{primary:string:enum(red,blue,green):Primary color}}, \
             notifications:boolean"]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        let settings = properties.get("settings").unwrap();
        assert_eq!(settings.r#type, Some(SchemaType::Object));

        let settings_props = settings.properties.as_ref().unwrap();
        let theme = settings_props.get("theme").unwrap();
        assert_eq!(theme.r#type, Some(SchemaType::Object));

        let theme_props = theme.properties.as_ref().unwrap();
        let mode = theme_props.get("mode").unwrap();
        assert_eq!(mode.r#type, Some(SchemaType::String));
        assert_eq!(mode.description, Some("Theme mode".to_string()));
        assert_eq!(
            mode.enum_values,
            Some(vec!["light".to_string(), "dark".to_string()])
        );

        let colors = theme_props.get("colors").unwrap();
        assert_eq!(colors.r#type, Some(SchemaType::Object));

        let colors_props = colors.properties.as_ref().unwrap();
        let primary = colors_props.get("primary").unwrap();
        assert_eq!(primary.r#type, Some(SchemaType::String));
        assert_eq!(primary.description, Some("Primary color".to_string()));
        assert_eq!(
            primary.enum_values,
            Some(vec![
                "red".to_string(),
                "blue".to_string(),
                "green".to_string()
            ])
        );
    }

    #[test]
    fn test_enum_edge_cases() {
        let func = FunctionDeclaration::new().with_parameters(&[
            // Empty enum values
            "status1, string:enum(), Status",
            // Single enum value
            "status2, string:enum(active), Status",
            // Enum values with special characters
            "status3, string:enum(in-progress,not_started,done!), Status",
            // Enum value containing parentheses
            "status4, string:enum((pending),(in-progress)), Status",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        let status1 = properties.get("status1").unwrap();
        assert_eq!(status1.enum_values, Some(vec![] as Vec<String>));

        let status2 = properties.get("status2").unwrap();
        assert_eq!(status2.enum_values, Some(vec!["active".to_string()]));

        let status3 = properties.get("status3").unwrap();
        assert_eq!(
            status3.enum_values,
            Some(vec![
                "in-progress".to_string(),
                "not_started".to_string(),
                "done!".to_string()
            ])
        );

        let status4 = properties.get("status4").unwrap();
        assert_eq!(
            status4.enum_values,
            Some(vec!["(pending)".to_string(), "(in-progress)".to_string()])
        );
    }

    #[test]
    fn test_complex_nested_objects() {
        let func = FunctionDeclaration::new().with_parameters(&[
            "config, object, Complex configuration | \
             database:{
                 connection:{
                     type:string:enum(mysql,postgres):Database type,
                     port:integer:Port number,
                     credentials:{
                         user:string,
                         password:string:Password for auth
                     }
                 },
                 settings:{
                     max_connections:integer,
                     timeout:integer:Connection timeout
                 }
             },
             logging:{
                 level:string:enum(debug,info,warn,error):Log level,
                 format:{
                     timestamp:boolean:Include timestamp,
                     source:boolean:Include source
                 }
             }",
        ]);

        let params = func.parameters.unwrap();
        let properties = params.properties;

        let config = properties.get("config").unwrap();
        assert_eq!(config.r#type, Some(SchemaType::Object));

        let config_props = config.properties.as_ref().unwrap();
        let database = config_props.get("database").unwrap();
        let db_props = database.properties.as_ref().unwrap();

        let connection = db_props.get("connection").unwrap();
        let conn_props = connection.properties.as_ref().unwrap();

        let db_type = conn_props.get("type").unwrap();
        assert_eq!(
            db_type.enum_values,
            Some(vec!["mysql".to_string(), "postgres".to_string()])
        );

        let credentials = conn_props.get("credentials").unwrap();
        let cred_props = credentials.properties.as_ref().unwrap();
        assert!(cred_props.contains_key("user"));
        assert!(cred_props.contains_key("password"));

        // Check logging configuration
        let logging = config_props.get("logging").unwrap();
        let log_props = logging.properties.as_ref().unwrap();

        let level = log_props.get("level").unwrap();
        assert_eq!(
            level.enum_values,
            Some(vec![
                "debug".to_string(),
                "info".to_string(),
                "warn".to_string(),
                "error".to_string()
            ])
        );

        let format = log_props.get("format").unwrap();
        let format_props = format.properties.as_ref().unwrap();
        assert!(format_props.contains_key("timestamp"));
        assert!(format_props.contains_key("source"));
    }
}
