use tokenizers::{Tokenizer, AddedToken};
use tokenizers::models::bpe::BPE;
use std::path::{Path, PathBuf};
use std::fs;
use std::error::Error;
use anyhow::{Result, Context};
use serde_json::Value;

pub struct TokenizerConfig {
    tokenizer_json: PathBuf,
    tokenizer_config_json: Option<PathBuf>,
    special_tokens_map_json: Option<PathBuf>,
    generation_config_json: Option<PathBuf>,
}

impl TokenizerConfig {
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let dir = dir.as_ref();
        let tokenizer_json = dir.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Err(Box::<dyn Error + Send + Sync>::from(format!("tokenizer.json not found in {:?}", dir)));
        }

        Ok(Self {
            tokenizer_json,
            tokenizer_config_json: Self::find_file(dir, "tokenizer_config.json"),
            special_tokens_map_json: Self::find_file(dir, "special_tokens_map.json"),
            generation_config_json: Self::find_file(dir, "generation_config.json"),
        })
    }

    fn find_file(dir: &Path, filename: &str) -> Option<PathBuf> {
        let path = dir.join(filename);
        path.exists().then_some(path)
    }
}

pub struct Tokenizer {
    tokenizer: Tokenizer,
    config: TokenizerConfig,
}

impl Tokenizer {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let config = TokenizerConfig::from_directory(model_dir)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(e))?;
        let tokenizer = Tokenizer::from_file(&config.tokenizer_json)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to load tokenizer from file: {:?}", e)))?;

        let mut llama_tokenizer = Self { tokenizer, config };
        llama_tokenizer.load_special_tokens()?;

        Ok(llama_tokenizer)
    }

    fn load_special_tokens(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(special_tokens_path) = &self.config.special_tokens_map_json {
            let content = fs::read_to_string(special_tokens_path)
                .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to read special_tokens_map.json: {:?}", e)))?;
            let special_tokens: Value = serde_json::from_str(&content)
                .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to parse special_tokens_map.json: {:?}", e)))?;

            for token_name in ["bos_token", "eos_token", "unk_token", "pad_token"] {
                if let Some(token) = special_tokens[token_name].as_str() {
                    let added_token = AddedToken::from(token.to_string(), true); // Convert &str to AddedToken
                    self.tokenizer.add_special_tokens(&[added_token]);
                }
            }
        }
        Ok(())
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let encoding = self.tokenizer.encode(text)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to encode text: {:?}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        self.tokenizer.decode(token_ids)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to decode tokens: {:?}", e)))
    }

    pub fn get_generation_config(&self) -> Result<Value, Box<dyn Error + Send + Sync>> {
        let content = self.config.generation_config_json
            .as_ref()
            .map(|path| fs::read_to_string(path))
            .transpose()
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to read generation_config.json: {:?}", e)))?;

        if let Some(content) = content {
            let value: Value = serde_json::from_str(&content)
                .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Failed to parse generation_config.json: {:?}", e)))?;
            Ok(value)
        } else {
            Err(Box::<dyn Error + Send + Sync>::from("No generation_config.json found"))
        }
    }
}
