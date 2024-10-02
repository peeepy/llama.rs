use crate::model::model::ModelConfig;
use std::error::Error;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::Result;  // Add this for better error handling
use tch::Kind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub head_dim: i64,
    pub hidden_act: String,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub num_key_value_heads: i64,
    pub pretraining_tp: i64,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    #[serde(with = "torch_dtype_serde")]
    pub torch_dtype: Kind,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: i64,
    pub rope_type: String,
}

mod torch_dtype_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use tch::Kind;

    pub fn serialize<S>(kind: &Kind, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = match kind {
            Kind::BFloat16 => "bfloat16",
            Kind::Float => "float32",
            Kind::Double => "float64",
            _ => return Err(serde::ser::Error::custom("Unsupported torch dtype")),
        };
        serializer.serialize_str(s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Kind, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "bfloat16" => Ok(Kind::BFloat16),
            "float32" => Ok(Kind::Float),
            "float64" => Ok(Kind::Double),
            _ => Err(serde::de::Error::custom("Unsupported torch dtype")),
        }
    }
}

impl ModelConfig for LlamaConfig {
    fn from_file(path: &Path) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let config_str = std::fs::read_to_string(path)?;
        let config: LlamaConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }
}

// impl LlamaConfig {
//     // Add a method to create a new LlamaConfig with default values
//     pub fn new() -> Self {
//         Self {
//             _name_or_path: String::new(),
//             architectures: vec!["LlamaForCausalLM".to_string()],
//             attention_bias: false,
//             attention_dropout: 0.0,
//             bos_token_id: 128000,
//             eos_token_id: 128001,
//             head_dim: 64,
//             hidden_act: "silu".to_string(),
//             hidden_size: 2048,
//             initializer_range: 0.02,
//             intermediate_size: 8192,
//             max_position_embeddings: 131072,
//             mlp_bias: false,
//             model_type: "llama".to_string(),
//             num_attention_heads: 32,
//             num_hidden_layers: 16,
//             num_key_value_heads: 8,
//             pretraining_tp: 1,
//             rms_norm_eps: 1e-05,
//             rope_scaling: RopeScaling {
//                 factor: 32.0,
//                 high_freq_factor: 4.0,
//                 low_freq_factor: 1.0,
//                 original_max_position_embeddings: 8192,
//                 rope_type: "llama3".to_string(),
//             },
//             rope_theta: 500000.0,
//             tie_word_embeddings: true,
//             torch_dtype: Kind::BFloat16,
//             transformers_version: "4.44.2".to_string(),
//             use_cache: true,
//             vocab_size: 128256,
//         }
//     }
// }

// impl Default for LlamaConfig {
//     fn default() -> Self {
//         Self {
//             hidden_size: 4096,
//             intermediate_size: 14336,
//             num_attention_heads: 32,
//             num_hidden_layers: 32,
//             num_key_value_heads: 8,
//             max_position_embeddings: 8192,
//             vocab_size: 128256,
//             rms_norm_eps: 1e-5,
//             bos_token_id: 128000,
//             eos_token_id: 128001,
//             pad_token_id: 128255,
//             rope_theta: 500000.0,
            
//             attention_bias: false,
//             attention_dropout: 0.0,
//             hidden_act: "silu".to_string(),
//             initializer_range: 0.02,
//             mlp_bias: false,
//             rope_scaling: None,
//             tie_word_embeddings: false,
//             torch_dtype: Kind::BFloat16,
//             use_cache: true,
//         }
//     }
// }