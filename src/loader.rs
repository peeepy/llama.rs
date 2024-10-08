// use std::error::Error;
// use std::path::Path;
// use std::fs::File;
// use safetensors::SafeTensors;
// use std::io::Read;

// pub fn load_model<P: AsRef<Path>>(dir: P) -> Result<Model, Box<dyn Error + Send + Sync>> {
//     let dir = dir.as_ref();  // Convert to &Path

//     let files = ModelFiles::from_directory(dir)?; // Use the same path

//     // Load the config and model from files
//     let config_str = std::fs::read_to_string(dir)?;
//     let config: Config = serde_json::from_str(&config_str)?;

//     let mut model_file = File::open(&files.model)?;
//     let mut buffer = Vec::new();
//     model_file.read_to_end(&mut buffer)?;
//     let safetensors = SafeTensors::deserialize(&buffer)?;

//     let model: Result<Model, Box<dyn Error + Send + Sync>> = Model::from_safetensors(&safetensors, config);
//     // Return the model instance
//     Ok(model?)
// }
