use crate::config::Config;
use crate::quantization::ModelTensor;
use crate::transformers::attention::Attention;
use crate::transformers::feed_forward::FeedForward;
use crate::transformers::rmsnorm::rmsnorm;
use crate::utils;
use rayon::iter::IntoParallelRefIterator;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::error::Error;
use rayon::iter::ParallelIterator;

pub struct TransformerBlock{
    pub attention: Attention,
    pub feed_forward: FeedForward,
    pub norm1_weight: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub epsilon: f32,
}

impl TransformerBlock {
    pub fn new(
        attention: Attention,
        feed_forward: FeedForward,
        norm1_weight: Vec<f32>,
        norm2_weight: Vec<f32>,
        epsilon: f32,
    ) -> Self {
        TransformerBlock {
            attention,
            feed_forward,
            norm1_weight,
            norm2_weight,
            epsilon,
        }
    }

    pub fn from_safetensors(
        st: &SafeTensors,
        config: Config,
        layer_index: usize,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Load attention components
        let wq = load_tensor(st, &format!("layers.{}.attention.wq.weight", layer_index))?;
        let wk = load_tensor(st, &format!("layers.{}.attention.wk.weight", layer_index))?;
        let wv = load_tensor(st, &format!("layers.{}.attention.wv.weight", layer_index))?;
        let wo = load_tensor(st, &format!("layers.{}.attention.wo.weight", layer_index))?;

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.quant_type(),
            config.group_size,
            config.dim / config.num_heads,
            config.num_heads,
        );

        // Load feed-forward components
        let w1 = load_tensor(st, &format!("layers.{}.feed_forward.w1.weight", layer_index))?;
        let w2 = load_tensor(st, &format!("layers.{}.feed_forward.w2.weight", layer_index))?;
        let w3 = load_tensor(st, &format!("layers.{}.feed_forward.w3.weight", layer_index))?;

        let feed_forward = FeedForward::new(
            w1,
            w2,
            w3,
            config.quant_type(),
            config.group_size,
            config.dim,
            config.hidden_dim,
        );

        // Load normalization weights
        let norm1_weight = load_tensor(st, &format!("layers.{}.attention_norm.weight", layer_index))?;
        let norm2_weight = load_tensor(st, &format!("layers.{}.ffn_norm.weight", layer_index))?;

        Ok(Self::new(
            attention,
            feed_forward,
            norm1_weight.flatten(0, -1).into(),
            norm2_weight.flatten(0, -1).into(),
            config.rms_norm_eps,
        ))
    }

    pub fn forward(
        &self,
        input: &ModelTensor,
        key_cache: &mut [f32],
        value_cache: &mut [f32],
        seq_len: usize,
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let dim = input.data.len();

        // Layer normalization before attention
        let mut normed_input = vec![0.0; dim];
        rmsnorm(
            &mut normed_input,
            &input.data,
            &self.norm1_weight,
            dim,
            self.epsilon,
        );

        // Attention mechanism
        let attention_output = self.attention.forward(
            &ModelTensor::new(&normed_input),
            key_cache,
            value_cache,
            seq_len,
            pos,
        )?;

        // Residual connection
        let mut residual_output = vec![0.0; dim];
        for i in 0..dim {
            residual_output[i] = input.data[i] + attention_output[i];
        }

        // Layer normalization before feed-forward
        let mut normed_residual = vec![0.0; dim];
        rmsnorm(
            &mut normed_residual,
            &residual_output,
            &self.norm2_weight,
            dim,
            self.epsilon,
        );

        // Feed-forward network
        let final_output = self
            .feed_forward
            .forward(&ModelTensor::new(&normed_residual))?;

        // Residual connection after feed-forward
        let mut output = vec![0.0; dim];
        for i in 0..dim {
            output[i] = residual_output[i] + final_output[i];
        }

        Ok(output)
    }
}

fn load_tensor(st: &SafeTensors, name: &str) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    let tensor_view = st.tensor(name)?;
    utils::from_tensor_view(tensor_view)
}
