use crate::model::model::{Model, ModelConfig};
use crate::model::model::LlamaModel;
use std::error::Error;
use std::path::Path;
use std::marker::PhantomData;

pub struct ModelLoader<M: Model> {
    _phantom: PhantomData<M>,
}

impl<M: Model> ModelLoader<M> {
    pub fn new() -> Self {
        ModelLoader {
            _phantom: PhantomData,
        }
    }

    pub fn load_model(model_path: &Path, config_path: &Path) -> Result<M, Box<dyn Error + Send + Sync>> {
        let config = M::Config::from_file(config_path)?;
        M::load_from_file(model_path, config)
    }
}

// Example usage for LlamaModel
pub type LlamaLoader = ModelLoader<LlamaModel>;