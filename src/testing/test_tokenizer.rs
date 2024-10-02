use std::path::Path;
use std::error::Error;
use crate::model::config::LlamaConfig;
use crate::model::model::{LlamaModel, Model, ModelConfig};
use crate::tokenizer::tokenizer::LlamaTokenizer;
use tch::{Tensor, Device};

pub fn test_tokenizer() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_dir = Path::new("D:/CODING/Rust/lm/Llama-3.2-1B");
    
    let start_time = std::time::Instant::now();
    let model = LlamaModel::load_from_directory(model_dir)?;
    // Initialize tokenizer using the same directory
    let tokenizer = LlamaTokenizer::new(model_dir)?;  // Use the same path
    let load_time = start_time.elapsed();
    
    println!("Model and tokenizer loaded in {:?}", load_time);

    let tokenization_start_time = std::time::Instant::now();
    let text = "Hello, world!";
    let encoded = tokenizer.encode(text)?;
    let decoded = tokenizer.decode(&encoded)?;
    let tokenization_time = start_time.elapsed();
    
    println!("Original text: {:?}", text);
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {:?}", decoded);
    println!("Time taken for tokenization test: {:?}", tokenization_time);
    Ok(())
}

