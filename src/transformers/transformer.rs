use crate::config::Config;
use crate::transformers::attention::Attention;
use crate::transformers::feed_forward::FeedForward;
use crate::transformers::transformer_block::{TransformerBlock, load_tensor};
// use crate::utils::{load_config, load_tensors_from_safetensors};
use safetensors::tensor::SafeTensors;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use tch::Tensor;
use glob::glob;
use std::sync::Arc;

pub struct Transformer {
    config: Config,
    token_embedding_table: Arc<Tensor>,
    blocks: Vec<TransformerBlock>,
}

impl Transformer {
    pub fn new(
        config: Config,
        token_embedding_table: Tensor, // This will be wrapped in Arc
        weights: Vec<Arc<Tensor>>,
    ) -> Self {
        let token_embedding_table = Arc::new(token_embedding_table);  // Wrap token_embedding in Arc
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let attention = Attention::new(
                Arc::clone(&weights[i * 4]),     // wq
                Arc::clone(&weights[i * 4 + 1]), // wk
                Arc::clone(&weights[i * 4 + 2]), // wv
                Arc::clone(&weights[i * 4 + 3]), // wo
                config.hidden_size / config.num_attention_heads, // head_dim
                config.num_attention_heads,  // number of heads
            );

            let feed_forward = FeedForward::new(
                Arc::clone(&weights[i * 3 + 4]), // w1
                Arc::clone(&weights[i * 3 + 5]), // w2
                Arc::clone(&weights[i * 3 + 6]), // w3
                config.hidden_size,              // hidden_size
                config.intermediate_size,         // intermediate_size (was hidden_dim)
            );

            let transformer_block = TransformerBlock::new(
                attention,
                feed_forward,
                vec![1.0; config.hidden_size],  // norm1 weights (placeholder)
                vec![1.0; config.hidden_size],  // norm2 weights (placeholder)
                config.rms_norm_eps,            // epsilon value for RMSNorm
            );

            blocks.push(transformer_block);
        }

        Transformer {
            config,
            token_embedding_table,  // Use Arc-wrapped tensor
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
    // Need to look at Tensor and possibly refactor fields
    pub fn forward(
    &mut self,
    input: &Tensor,
    key_cache: &mut Vec<f32>,
    value_cache: &mut Vec<f32>,
    seq_len: usize,
    pos: usize,
) -> Tensor {
    // Shallow clone the input tensor so we can modify it in-place
    let mut output = input.shallow_clone();

    // Iterate over transformer blocks and update the output tensor
    for block in &mut self.blocks {
        output = block
            .forward(
                &output,        // Pass the output Tensor from the previous block
                key_cache,      // Mutable key cache reference
                value_cache,    // Mutable value cache reference
                seq_len,
                pos,
            )
            .unwrap();        // Unwrap the Result to get the Tensor output
    }

    output // Return the final output tensor
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
//         let input_tensor = Tensor::new(&input_ids);

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
