use crate::config::Config; 
use crate::transformers::attention::Attention;
use crate::transformers::feed_forward::FeedForward;
use crate::transformers::rmsnorm::rmsnorm;
// use crate::utils;
use tch::Kind;
use std::error::Error;
use safetensors::SafeTensors;
use tch::Tensor;
use std::sync::Arc;

pub struct TransformerBlock {
    pub attention: Attention,
    pub feed_forward: FeedForward,
    pub norm1_weight: Tensor,
    pub norm2_weight: Tensor,
    pub epsilon: f32,
}

impl TransformerBlock {
    pub fn new(
        attention: Attention,
        feed_forward: FeedForward,
        norm1_weight: Tensor,
        norm2_weight: Tensor,
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
    let wq: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.self_attn.q_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).transpose_(0,1).into();
    let wk: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.self_attn.k_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).transpose_(0,1).into();
    let wv: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.self_attn.v_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).transpose_(0,1).into();
    let wo: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.self_attn.o_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).transpose_(0,1).into();

    // Debugging: Print shapes
    for name in st.names() {
    let tensor = load_tensor(st, name)?;
    println!("Tensor name: {}, shape: {:?}", name, tensor.size());
}

    let attention = Attention::new(
        Arc::clone(&wq),
        Arc::clone(&wk),
        Arc::clone(&wv),
        Arc::clone(&wo),
        config.get_head_dim(),
        config.num_attention_heads,
    );

    // Load feed-forward components
    let w1: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.mlp.gate_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).into();
    let w2: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.mlp.down_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).into();
    let w3: Arc<Tensor> = load_tensor(st, &format!("model.layers.{}.mlp.up_proj.weight", layer_index))?
        .to_kind(tch::Kind::Float).into();

    // Debugging: Print feedforward weights shapes
    println!("mlp.gate_proj.weight shape: {:?}", w1.size());
    println!("mlp.down_proj.weight shape: {:?}", w2.size());
    println!("mlp.up_proj.weight shape: {:?}", w3.size());
    let feed_forward = FeedForward::new(
        Arc::clone(&w1),
        Arc::clone(&w2),
        Arc::clone(&w3),
        config.get_head_dim(),
        config.intermediate_size,
    );

    // Load normalization weights
    let norm1_weight = load_tensor(st, &format!("model.layers.{}.input_layernorm.weight", layer_index))?
        .to_kind(tch::Kind::Float);
    let norm2_weight = load_tensor(st, &format!("model.layers.{}.post_attention_layernorm.weight", layer_index))?
        .to_kind(tch::Kind::Float);

    // [2048]. This is implicitly [1, 1, 2048] so is compatible with other tensors
    println!("norm1_weight shape: {:?}", norm1_weight.size());
    println!("norm2_weight shape: {:?}", norm2_weight.size());

    Ok(Self::new(
        attention,
        feed_forward,
        norm1_weight,
        norm2_weight,
        config.rms_norm_eps,
    ))
}


pub fn forward(
    &self,
    input: &Tensor, // Input should be [batch_size, seq_len, embedding_dim]
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    pos: usize,
) -> Result<Tensor, Box<dyn Error + Sync + Send>> {
    // Extract dimensions
    let batch_size = input.size()[0];
    let seq_len = input.size()[1];
    let dim = input.size()[2]; // Embedding dimension

    // Debugging: Print input shape and size
    println!("TransformerBlock input shape: {:?}", input.size());
    println!("Batch size: {:?}", batch_size);
    println!("Sequence length: {:?}", seq_len);
    println!("Embedding dimension: {:?}", dim);

    // Layer normalization before attention
    let normed_input = rmsnorm(input, &self.norm1_weight, self.epsilon)?;
    println!("Normed input tensor shape: {:?}", normed_input.size());

    // Proceed with the attention forward pass
    let attention_output = self.attention.forward(
        Arc::new(normed_input.clone(&normed_input)),
        key_cache,
        value_cache,
        pos,
    )?;

    println!("Attention output shape: {:?}", attention_output.size());

    // Residual connection: Add attention output to the original input
    let residual_output = input + attention_output.as_ref();
    println!("Residual output shape: {:?}", residual_output.size());

    // Layer normalization before feed-forward
    let normed_residual = rmsnorm(&residual_output, &self.norm2_weight, self.epsilon)?;
    println!("Normed residual shape: {:?}", normed_residual.size());

    // Feed-forward network
    let feedforward_output = self.feed_forward.forward(Arc::new(normed_residual))?;
    println!("Feed-forward final output shape: {:?}", feedforward_output.size());

    // Final residual connection: Add feedforward output to residual output
    let final_output = residual_output + feedforward_output.as_ref();
    println!("Final output shape: {:?}", final_output.size());

    // Return the final output tensor
    Ok(final_output)
}





}


pub fn load_tensor(st: &SafeTensors, name: &str) -> Result<Arc<Tensor>, Box<dyn Error + Send + Sync>> {
    let tensor_view = st.tensor(name)?;
    
    let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
    let dtype = match tensor_view.dtype() {
        safetensors::Dtype::F32 => Kind::Float,
        safetensors::Dtype::BF16 => Kind::BFloat16,
        safetensors::Dtype::F16 => Kind::Half,
        safetensors::Dtype::I8 => Kind::Int8,
        _ => return Err("Unsupported dtype".into()),
    };

    let raw_data = tensor_view.data();
    let tensor = match dtype {
        Kind::Float => Tensor::f_from_slice(bytemuck::cast_slice::<u8, f32>(raw_data))?,
        Kind::BFloat16 => {
            let f32_data: Vec<f32> = bytemuck::cast_slice::<u8, u16>(raw_data)
                .iter()
                .map(|&x| f32::from_bits((x as u32) << 16))
                .collect();
            Tensor::f_from_slice(&f32_data)?.to_kind(Kind::BFloat16)
        },
        Kind::Half => {
            let f32_data: Vec<f32> = bytemuck::cast_slice::<u8, u16>(raw_data)
                .iter()
                .map(|&x| {
                    let sign = ((x & 0x8000) as u32) << 16;
                    let exponent = (((x & 0x7c00) >> 10) as u32) << 23;
                    let mantissa = ((x & 0x03ff) as u32) << 13;
                    f32::from_bits(sign | exponent | mantissa)
                })
                .collect();
            Tensor::f_from_slice(&f32_data)?.to_kind(Kind::Half)
        },
        Kind::Int8 => Tensor::f_from_slice(raw_data)?,
        _ => unreachable!(),
    };

    Ok(Arc::new(tensor.reshape(&shape)))
}
