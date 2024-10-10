use std::error::Error;
use std::path::Path;
use crate::inference;
use crate::sampler::Sampler;
use crate::transformers::transformer::Transformer;
use crate::tokenizer::hf_tokenizer::ModelTokenizer;

pub fn test_tokenizer() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_dir = Path::new("D:/CODING/Rust/lm/Llama-3.2-1B");

    let start_time = std::time::Instant::now();
    println!("#### MODEL LOADING ####");
    let mut model: Transformer = Transformer::load_from_directory(model_dir)?;
    // // Initialize tokenizer using the same directory
    let mut tokenizer = ModelTokenizer::new(model_dir)?; // Use the same path
    let load_time = start_time.elapsed();

    println!("Model & Tokenizer loaded in {:?}", load_time);

    println!("#### TOKENIZATION TEST ####");
    let tokenization_start_time = std::time::Instant::now();
    let text = "Hello, world!";
    let encoded = tokenizer.encode(text)?;
    let decoded = tokenizer.decode(&encoded)?;
    let tokenization_time = tokenization_start_time.elapsed();

    println!("Original text: {:?}", text);
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {:?}", decoded);
    println!("Time taken for tokenization test: {:?}", tokenization_time);

    println!("#### INFERENCE TEST ####");
    let mut sampler = Sampler::new(model.config.vocab_size as u32, 0.7, 0.5, 1);
    let start_time = std::time::Instant::now();
    let generated_text = inference::simple_inference(&mut model,&mut tokenizer, &mut sampler,"Hello world ")?;
    println!("Hello {:?}", generated_text);
    let inference_time = start_time.elapsed();
    println!("Inferenced successfully in {:?}", inference_time);
    Ok(())
}
