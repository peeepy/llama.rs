use crate::utils::softmax;
use std::error::Error;
use tch::Tensor;
use std::sync::Arc;

pub struct Attention {
    pub wq: Arc<Tensor>,
    pub wk: Arc<Tensor>,
    pub wv: Arc<Tensor>,
    pub wo: Arc<Tensor>,
    pub head_dim: usize,
    pub num_heads: usize,
}

impl Attention {
    pub fn new(
        wq: Arc<Tensor>,
        wk: Arc<Tensor>,
        wv: Arc<Tensor>,
        wo: Arc<Tensor>,
        head_dim: usize,
        num_heads: usize,
    ) -> Self {
        Attention {
            wq,
            wk,
            wv,
            wo,
            head_dim,
            num_heads,
        }
    }

    pub fn forward(
        &self,
        input: Arc<Tensor>,
        key_cache: &mut Vec<f32>,
        value_cache: &mut Vec<f32>,
        seq_len: usize,
        pos: usize,
    ) -> Result<Arc<Tensor>, Box<dyn Error>> {
        // Compute queries, keys, and values
        let queries: Tensor = input.matmul(&self.wq);
        let keys: Tensor = input.matmul(&self.wk);
        let values: Tensor = input.matmul(&self.wv);

        // Convert tensors to Vec<f32> and store in the cache
        let keys_vec: Vec<f32> = Vec::<f32>::try_from(keys)?;
        let values_vec: Vec<f32> = Vec::<f32>::try_from(values)?;

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
        let mut output: Vec<f32> = vec![0.0; self.head_dim];
        for t in 0..seq_len {
            let cache_values = &value_cache[t * self.head_dim..(t + 1) * self.head_dim];
            let score: f32 = attention_scores[t];
            for i in 0..self.head_dim {
                output[i] += score * cache_values[i];
            }
        }

        // Convert `output` from Vec<f32> to Tensor
        let output_tensor: Tensor = Tensor::from_slice(&output);

        // Apply output projection using matmul and handle Result
        let final_output_tensor = output_tensor.matmul(&self.wo);

        Ok(Arc::new(final_output_tensor))  // Wrap the result in Arc<Tensor>
    }

}