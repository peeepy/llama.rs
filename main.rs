use log::{info, error};
use lm::testing::test_loading::test_load_model;
use lm::testing::test_tokenizer::test_tokenizer;

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