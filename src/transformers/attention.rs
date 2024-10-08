// attention.rs
use crate::tensor::{ModelTensor, QuantType};
use crate::utils::softmax;
use std::error::Error;
use tch::Tensor;

pub struct Attention {
    pub wq: ModelTensor,
    pub wk: ModelTensor,
    pub wv: ModelTensor,
    pub wo: ModelTensor,
    pub quant_type: QuantType,
    pub group_size: usize,
    pub head_dim: usize,
    pub num_heads: usize,
}

impl Attention {
    pub fn forward(
    &self,
    input: &ModelTensor,
    key_cache: &mut Vec<f32>,
    value_cache: &mut Vec<f32>,
    seq_len: usize,
    pos: usize,
) -> Result<Tensor, Box<dyn Error>> {
        // Compute queries, keys, and values
        let queries: Tensor = input.data.matmul(&self.wq.data);
        let keys: Tensor = input.data.matmul(&self.wk.data);
        let values: Tensor = input.data.matmul(&self.wv.data);

        // Convert tensors to vec and store in the cache
        // Convert tensors to vec and store in the cache
        let keys_vec: Vec<f32> = Vec::<f32>::try_from(keys).expect("Failed to convert keys to Vec<f32>");
        let values_vec: Vec<f32> = Vec::<f32>::try_from(values).expect("Failed to convert values to Vec<f32>");

        // Ensure the vectors have the correct length
        assert_eq!(keys_vec.len(), self.head_dim, "Keys vector has incorrect length");
        assert_eq!(values_vec.len(), self.head_dim, "Values vector has incorrect length");

        // Copy the vectors into the cache
        key_cache[pos * self.head_dim..(pos + 1) * self.head_dim].copy_from_slice(&keys_vec);
        value_cache[pos * self.head_dim..(pos + 1) * self.head_dim].copy_from_slice(&values_vec);

        // Compute attention scores
        let mut attention_scores: Vec<f32> = vec![0.0; seq_len];
        for t in 0..seq_len {
            let cache_keys: &[f32] = &key_cache[t * self.head_dim..(t + 1) * self.head_dim];
            let mut score: f32 = 0.0;
            for i in 0..self.head_dim {
                // Convert the f64 from double_value to f32 before multiplication
                score += (queries.double_value(&[i as i64]) as f32) * cache_keys[i];
            }
            score /= (self.head_dim as f32).sqrt();
            attention_scores[t] = score;
        }

        // Apply softmax to the attention scores
        softmax(&mut attention_scores);

        // Compute the weighted sum of values based on attention scores
        let mut output = vec![0.0; self.head_dim];
        for t in 0..seq_len {
            let cache_values = &value_cache[t * self.head_dim..(t + 1) * self.head_dim];
            let score = attention_scores[t];
            for i in 0..self.head_dim {
                output[i] += score * cache_values[i];
            }
        }

        // Convert `output` from Vec<f32> to tch::Tensor
        let output_tensor: tch::Tensor = tch::Tensor::f_from_slice(&output)?;

        // Apply output projection using matmul and handle Result
        let final_output_tensor = output_tensor.matmul(&self.wo.data);

        Ok(final_output_tensor)
    }

    pub fn from_safetensors(
        st: &SafeTensors,
        layer_index: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // Load the tensors (wq, wk, wv, wo) for attention
        let wq = ModelTensor::from_safetensors(st, format!("wq_{}", layer_index))?;
        let wk = ModelTensor::from_safetensors(st, format!("wk_{}", layer_index))?;
        let wv = ModelTensor::from_safetensors(st, format!("wv_{}", layer_index))?;
        let wo = ModelTensor::from_safetensors(st, format!("wo_{}", layer_index))?;
        
        Ok(Attention {
            wq,
            wk,
            wv,
            wo,
            quant_type: QuantType::None, // Update as needed
            group_size: 32, // Example group size
            head_dim: 64, // Example head dimension
            num_heads: 8, // Example number of heads
        })
    }
}
