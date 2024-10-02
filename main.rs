use log::{info, error};
use lm::testing::test_loading::test_load_model;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::init();

    info!("Starting model loading...");
    match test_load_model() {
        Ok(_) => {
            info!("Model loaded successfully");
            Ok(())
        },
        Err(e) => {
            error!("Failed to load model: {}", e);
            Err(e)
        }
    }
}