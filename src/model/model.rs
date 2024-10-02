use crate::model::config::LlamaConfig;
use crate::model::files::ModelFiles;
use crate::tokenizer::tokenizer::LlamaTokenizer;
use crate::quantization::quantized_weights::{QuantizedTensor, QuantType, TensorData};
use safetensors::{SafeTensors, tensor::TensorView};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use tch::{Tensor, Kind, Device};
use rayon::prelude::*;

pub trait ModelConfig: Sized + Send + Sync {
    fn from_file(path: &Path) -> Result<Self, Box<dyn Error + Send + Sync>>;
}

pub trait Model: Sized {
    type Config: ModelConfig;

    fn load_from_file(model_path: &Path, config: Self::Config) -> Result<Self, Box<dyn Error + Send + Sync>>;
}

pub struct LlamaModel {
    pub config: LlamaConfig,
    pub tensors: HashMap<String, QuantizedTensor>,
}

impl LlamaModel {
    pub fn new(config: LlamaConfig) -> Self {
        Self {
            config,
            tensors: HashMap::new(),
        }
    }

    pub fn load_from_directory<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let dir = dir.as_ref();  // Convert to &Path

        let files = ModelFiles::from_directory(dir)?; // Use the same path

        // Load the config and model from files
        let config = LlamaConfig::from_file(&files.config)?;
        let llama_model = Self::load_from_file(&files.model, config)?;

        // Return the LlamaModel instance (no tokenizer inside LlamaModel struct)
        Ok(llama_model)
    }



    pub fn quantize(&mut self, quant_type: QuantType, group_size: usize) {
        for (_, tensor) in self.tensors.iter_mut() {
            if let TensorData::Unquantized(data) = &tensor.data {
                *tensor = QuantizedTensor::new(data, quant_type, group_size);
            }
        }
    }

    pub fn from_safetensors(st: &SafeTensors, config: LlamaConfig) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tensors: HashMap<String, QuantizedTensor> = st.names()
            .par_iter()
            .map(|name| {
                let tensor_view = st.tensor(name)
                    .map_err(|e| Box::<dyn Error + Send + Sync>::from(format!("Error loading tensor '{}': {}", name, e)))?;
                let tensor = Self::from_tensor_view_unquantized(tensor_view)?;
                let tensor_slice: Vec<f32> = tensor.flatten(0, -1).into();
                let quantized_tensor = QuantizedTensor::new(&tensor_slice, QuantType::None, 1);
                Ok((name.to_string(), quantized_tensor))
            })
            .collect::<Result<_, Box<dyn Error + Send + Sync>>>()?;

        Ok(Self { config, tensors })
    }

    pub fn from_tensor_view_unquantized(view: TensorView) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        let shape: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
        let dtype = match view.dtype() {
            safetensors::Dtype::BF16 => Kind::BFloat16,
            safetensors::Dtype::I8 => Kind::Int8,
            _ => return Err("Unsupported dtype for unquantized loading.".into()),
        };

        let raw_data = view.data();
        let tensor = match dtype {
            Kind::BFloat16 => {
                let f32_data: Vec<f32> = bytemuck::cast_slice::<u8, u16>(raw_data)
                    .iter()
                    .map(|&x| {
                        let bits = x as u32;
                        let sign = bits >> 15;
                        let exp = (bits >> 7) & 0xFF;
                        let mantissa = bits & 0x7F;
                        let f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 16);
                        f32::from_bits(f32_bits)
                    })
                    .collect();
                Tensor::of_slice(&f32_data).to_dtype(Kind::BFloat16, false, false)
            },
            Kind::Int8 => Tensor::of_slice(raw_data),
            _ => unreachable!(),
        };

        Ok(tensor.reshape(&shape))
    }
}

impl Model for LlamaModel {
    type Config = LlamaConfig;

    fn load_from_file(model_path: &Path, config: LlamaConfig) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let safetensors = SafeTensors::deserialize(&buffer)?;

        Self::from_safetensors(&safetensors, config)
    }
}