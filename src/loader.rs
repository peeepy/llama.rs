use crate::model::config::Config;
use crate::model::model::Model;
use std::error::Error;
use std::path::Path;
use std::marker::PhantomData;

pub fn load_model<P: AsRef<Path>>(dir: P) -> Result<Model, Box<dyn Error + Send + Sync>> {
    let dir = dir.as_ref();  // Convert to &Path

    let files = ModelFiles::from_directory(dir)?; // Use the same path

    // Load the config and model from files
    let config_str = std::fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&config_str)?;

    let mut model_file = File::open(&files.model)?;
    let mut buffer = Vec::new();
    safetensors_file.read_to_end(&mut buffer)?;
    let safetensors = SafeTensors::deserialize(&buffer)?;

    let model: Result<Model, Box<dyn Error + Send + Sync>> = Model::from_safetensors(&safetensors, config);
    // Return the LlamaModel instance (no tokenizer inside LlamaModel struct)
    Ok(model)
}