use std::path::Path;
use std::error::Error;
use crate::model::config::LlamaConfig;
use crate::model::model::{LlamaModel, Model, ModelConfig};
use tch::{Tensor, Device};

pub fn test_load_model() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_dir = Path::new("D:/CODING/Rust/lm/Llama-3.2-1B");
    
    let start_time = std::time::Instant::now();
    let model = LlamaModel::load_from_directory(model_dir)?;
    let load_time = start_time.elapsed();
    println!("Model loaded successfully in {:?}", load_time);

    println!("Model config: {:?}", model.config);
    // Check tensor information
    for (name, tensor) in model.tensors {
        println!("Tensor shape: {:?} - Quantization: {:?}", tensor.original_shape, tensor.quant_type);
    }

    Ok(())
}
