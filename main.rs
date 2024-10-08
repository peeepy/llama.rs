use lm::testing::test_loading::test_load_model;
use lm::testing::test_tokenizer::test_tokenizer;
use log::{error, info};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::init();

    info!("Starting model loading...");
    test_load_model()?;
    info!("Model loaded successfully");

    info!("Beginning tokenizer test...");
    test_tokenizer()?;
    info!("Tokenization test succeeded");

    Ok(())
}

// NOTE: suggested full usage example of new mods, not fully aligned with my goals
// fn main() -> Result<(), Box<dyn Error>> {
//     // Load the configuration from the model directory
//     let config = ModelConfig::load("path/to/config.json")?;

//     // Load the weights from the safetensors file
//     let weights = load_weights("path/to/model.safetensors");

//     // Initialize the transformer model
//     let mut transformer = Transformer::new(config, weights);

//     // Load the tokenizer
//     let vocab = HashMap::from([("hello".to_string(), 1), ("world".to_string(), 2), ("<EOS>".to_string(), 0)]);
//     let tokenizer = Tokenizer::new(vocab, 0);

//     // Initialize the sampler (with top-k and temperature sampling)
//     let sampler = Sampler::new(Some(5), None, 1.0); // Top-k with k=5, temperature=1.0

//     // Generate text
//     let prompt = "hello";
//     let generated_text = transformer.generate(&tokenizer, &sampler, prompt, 50)?;

//     println!("Generated Text: {}", generated_text);

//     Ok(())
// }
