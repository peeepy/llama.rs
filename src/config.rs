use anyhow::Result; // Add this for better error handling
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;
use tch::Kind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
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

    // pub fn serialize<S>(kind: &Kind, serializer: S) -> Result<S::Ok, S::Error>
    // where
    //     S: Serializer,
    // {
    //     let s = match kind {
    //         Kind::BFloat16 => "bfloat16",
    //         Kind::Float => "float32",
    //         Kind::Double => "float64",
    //         _ => return Err(serde::ser::Error::custom("Unsupported torch dtype")),
    //     };
    //     serializer.serialize_str(s)
    // }

    // pub fn deserialize<'de, D>(deserializer: D) -> Result<Kind, D::Error>
    // where
    //     D: Deserializer<'de>,
    // {
    //     let s = String::deserialize(deserializer)?;
    //     match s.as_str() {
    //         "bfloat16" => Ok(Kind::BFloat16),
    //         "float32" => Ok(Kind::Float),
    //         "float64" => Ok(Kind::Double),
    //         _ => Err(serde::de::Error::custom("Unsupported torch dtype")),
    //     }
    // }
}
