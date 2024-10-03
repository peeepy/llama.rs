use std::path::Path;
use std::error::Error;
use crate::{loader, model::config::Config};
use crate::model::model::Model;
use crate::loader::load_model;
use tch::{Tensor, Device};

pub fn test_load_model() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_dir = Path::new("D:/CODING/Rust/lm/Llama-3.2-1B");
    
    let start_time = std::time::Instant::now();
    let model = loader::load_model(model_dir)?;
    let load_time = start_time.elapsed();
    println!("Model loaded successfully in {:?}", load_time);
    // You can add more detailed prints here if needed
    println!("Model config: {:?}", model.config);
    // println!("Number of tensors: {}", model.tensors.len());

    // Check tensor information
    for (name, tensor) in model.tensors {
        println!("Tensor shape: {:?} - Quantization: {:?}", tensor.original_shape, tensor.quant_type);
    }

    Ok(())
}