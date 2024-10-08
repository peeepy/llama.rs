use crate::config::Config;
use crate::transformers::attention::Attention;
use crate::transformers::feed_forward::FeedForward;
use crate::transformers::rmsnorm::rmsnorm;
use crate::utils;
use std::error::Error;
use safetensors::SafeTensors;
use tch::Tensor;
use std::sync::Arc;

pub struct TransformerBlock {
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
            Arc::clone(&wq),
            Arc::clone(&wk),
            Arc::clone(&wv),
            Arc::clone(&wo),
            config.get_head_dim(),
            config.num_attention_heads,
        );

        // Load feed-forward components
        let w1 = load_tensor(st, &format!("layers.{}.feed_forward.w1.weight", layer_index))?;
        let w2 = load_tensor(st, &format!("layers.{}.feed_forward.w2.weight", layer_index))?;
        let w3 = load_tensor(st, &format!("layers.{}.feed_forward.w3.weight", layer_index))?;

        let feed_forward = FeedForward::new(
            Arc::clone(&w1),
            Arc::clone(&w2),
            Arc::clone(&w3),
            config.get_head_dim(),
            config.intermediate_size,
        );

        // Load normalization weights
        let norm1_weight = load_tensor(st, &format!("layers.{}.attention_norm.weight", layer_index))?;
        let norm2_weight = load_tensor(st, &format!("layers.{}.ffn_norm.weight", layer_index))?;

        Ok(Self::new(
            attention,
            feed_forward,
            Vec::<f32>::try_from(norm1_weight.flatten(0, -1))?,
            Vec::<f32>::try_from(norm2_weight.flatten(0, -1))?,
            config.rms_norm_eps,
        ))
    }

    pub fn forward(
        &self,
        input: &Tensor,
        key_cache: &mut Vec<f32>,
        value_cache: &mut Vec<f32>,
        seq_len: usize,
        pos: usize,
    ) -> Result<Tensor, Box<dyn Error>> {
        let dim = input.size1()? as usize;

        // Convert input tensor to Vec<f32>
        let input_data: Vec<f32> = Vec::<f32>::try_from(input)?;

        // Layer normalization before attention
        let mut normed_input = vec![0.0; dim];
        rmsnorm(
            &mut normed_input,
            &input_data,  // passing Vec<f32> reference
            &self.norm1_weight,
            dim,
            self.epsilon,
        );

        // Convert normed_input (Vec<f32>) to Tensor
        let normed_input_tensor = Tensor::from_slice(&normed_input);

        // Attention mechanism with Tensor
        let attention_output = self.attention.forward(
            Arc::new(normed_input_tensor),
            key_cache,
            value_cache,
            seq_len,
            pos,
        )?;

        // Convert attention_output (Tensor) to Vec<f32> for residual connection
        let attention_output_vec: Vec<f32> = Vec::<f32>::try_from(attention_output)?;

        // Residual connection
        let mut residual_output = vec![0.0; dim];
        for i in 0..dim {
            residual_output[i] = input_data[i] + attention_output_vec[i]; // input_data from earlier conversion
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

        // Convert normed_residual (Vec<f32>) to Tensor
        let normed_residual_tensor = Tensor::from_slice(&normed_residual);

        // Feed-forward network
        let final_output_tensor = self.feed_forward.forward(Arc::new(normed_residual_tensor))?;

        // Convert final_output (Tensor) to Vec<f32> for final residual connection
        let final_output_vec: Vec<f32> = Vec::<f32>::try_from(final_output_tensor)?;

        // Residual connection after feed-forward
        let mut output = vec![0.0; dim];
        for i in 0..dim {
            output[i] = residual_output[i] + final_output_vec[i];
        }

        // Convert final output (Vec<f32>) back to Tensor before returning
        Ok(Tensor::from_slice(&output))
    }
}

pub fn load_tensor(st: &SafeTensors, name: &str) -> Result<Arc<Tensor>, Box<dyn Error + Send + Sync>> {
    // Access the Ok(TensorView)
    let tensor_view = match st.tensor(name) {
        Err(e) => return Err(Box::new(e)),  // Box the SafeTensorError and propagate
        Ok(tensor_view) => tensor_view,  // Unwrap TensorView
    };

    // Now pass the unwrapped TensorView into utils::from_tensor_view
    let returned_tensor = match utils::from_tensor_view(tensor_view) {
        Err(e) => return Err(e),  // Propagate the error from from_tensor_view
        Ok(returned_tensor) => returned_tensor,  // Unwrap the Ok value to get the tch::Tensor
    };

    // Now you have `returned_tensor` as a `tch::Tensor`
    Ok(Arc::new(returned_tensor))
}
