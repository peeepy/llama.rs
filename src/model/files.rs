use std::path::{Path, PathBuf};
use anyhow::{Result, Context};

pub struct ModelFiles {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer_json: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    pub special_tokens_map: Option<PathBuf>,
    pub generation_config: Option<PathBuf>,
}

impl ModelFiles {
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();

        let model = Self::find_file(dir, &["*.safetensors"])
            .context("Model file not found")?;
        let config = Self::find_file(dir, &["config.json"])
            .context("Config file not found")?;
        let tokenizer_json = Self::find_file(dir, &["tokenizer.json"])
            .context("Tokenizer JSON file not found")?;

        Ok(Self {
            model,
            config,
            tokenizer_json,
            tokenizer_config: Self::find_file(dir, &["tokenizer_config.json"]),
            special_tokens_map: Self::find_file(dir, &["special_tokens_map.json"]),
            generation_config: Self::find_file(dir, &["generation_config.json"]),
        })
    }

    fn find_file(dir: &Path, possible_names: &[&str]) -> Option<PathBuf> {
        possible_names.iter()
            .map(|name| dir.join(name))
            .find(|path| path.exists())
    }
}