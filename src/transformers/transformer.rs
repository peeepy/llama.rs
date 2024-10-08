use crate::config::Config;
use crate::quantization::ModelTensor;
use crate::transformers::attention::Attention;
use crate::transformers::feed_forward::FeedForward;
use crate::transformers::transformer_block::TransformerBlock;
// use crate::utils::{load_config, load_tensors_from_safetensors};
use safetensors::tensor::SafeTensors;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use tch::Tensor;

// ERROR: expected 0 lifetime arguments
// Need to look into ModelTensor again
pub struct Transformer {
    pub config: Config,
    pub token_embedding_table: Tensor, // Embeddings stored directly in Transformer
    pub blocks: Vec<TransformerBlock>, // Transformer consists of multiple blocks
}

impl Transformer {
    pub fn new(config: Config, token_embedding_table: Tensor, weights: Vec<Tensor>) -> Self {
        let quant_type = config.quant_type();
        let group_size = config.group_size;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let attention = Attention::new(
                weights[i * 4],     // wq
                weights[i * 4 + 1], // wk
                weights[i * 4 + 2], // wv
                weights[i * 4 + 3], // wo
                quant_type,
                group_size,
                config.dim / config.n_heads, // head_dim
                config.n_heads,
            );

            let feed_forward = FeedForward::new(
                weights[i * 3 + 4], // w1
                weights[i * 3 + 5], // w2
                weights[i * 3 + 6], // w3
                quant_type,
                group_size,
                config.dim,
                config.hidden_dim,
            );

            let transformer_block = TransformerBlock::new(
                attention,
                feed_forward,
                vec![1.0; config.dim], // norm1 weights (placeholder, load actual weights here)
                vec![1.0; config.dim], // norm2 weights (placeholder, load actual weights here)
                config.rms_norm_eps,
            );

            blocks.push(transformer_block);
        }

        Transformer {
            config,
            token_embedding_table,
            blocks,
        }
    }

    pub fn load_from_directory<P: AsRef<std::path::Path>>(
        dir: P,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let dir = dir.as_ref();

        // Load config.json for model configuration
        let config = load_config(dir.join("config.json"))?;

        // Find and load the SafeTensors file
        let model_file = format!("{}/{}", dir.display(), "*.safetensors");
        let model_path = glob(&model_file)?
            .filter_map(Result::ok)
            .next()
            .ok_or("No SafeTensors file found")?;

        let mut safetensor_file = File::open(model_path)?;
        let mut buffer = Vec::new();
        safetensor_file.read_to_end(&mut buffer)?;
        let safetensors = SafeTensors::deserialize(&buffer)?;

        // Load the token embedding table
        let token_embedding_table = load_tensor(&safetensors, "token_embedding_table")?;

        // Load transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let block = TransformerBlock::from_safetensors(&safetensors, config.clone(), i)?;
            blocks.push(block);
        }

        Ok(Transformer {
            config,
            token_embedding_table,
            blocks,
        })
    }

    // ERROR:  expected struct `Vec<f32>`
    // found enum `QuantState`
    // Need to look at ModelTensor and possibly refactor fields
    pub fn forward(
        &mut self,
        input: &ModelTensor,
        key_cache: &mut [f32],
        value_cache: &mut [f32],
        seq_len: usize,
        pos: usize,
    ) -> Vec<f32> {
        let mut output = input.data.clone();
        for block in &mut self.blocks {
            output = block
                .forward(
                    &ModelTensor::new(&output),
                    key_cache,
                    value_cache,
                    seq_len,
                    pos,
                )
                .unwrap();
        }
        output
    }
}

// NOTE: probably won't put it here. unnecessary abstraction?
//     pub fn generate(
//     &mut self,
//     tokenizer: &Tokenizer,
//     sampler: &Sampler,
//     prompt: &str,
//     max_length: usize,
// ) -> Result<String, Box<dyn Error>> {
//     // Tokenize the prompt
//     let mut input_ids = tokenizer.tokenize(prompt);

//     // Initialize key and value caches
//     let mut key_cache = vec![0.0; self.config.n_layers * self.config.seq_len * self.config.hidden_dim];
//     let mut value_cache = vec![0.0; self.config.n_layers * self.config.seq_len * self.config.hidden_dim];

//     // Start generating tokens
//     for pos in 0..max_length {
//         // Create input tensor from token IDs
//         let input_tensor = ModelTensor::new(&input_ids);

//         // Forward pass through the transformer to get logits
//         let logits = self.forward(&input_tensor, &mut key_cache, &mut value_cache, self.config.seq_len, pos);

//         // Use the sampler to pick the next token based on logits
//         let next_token_id = sampler.sample(&mut logits);

//         // Stop if we hit the end of sequence token
//         if next_token_id == tokenizer.eos_token_id() {
//             break;
//         }

//         // Append the new token to the input
//         input_ids.push(next_token_id);
//     }

//     // Detokenize the final sequence of token IDs to generate the text
//     let output_text = tokenizer.detokenize(&input_ids);

//     Ok(output_text)
// }
// }
