use crate::config::Config;
use crate::transformers::attention::Attention;
use crate::transformers::feed_forward::FeedForward;
use crate::transformers::transformer_block::{TransformerBlock, load_tensor};
use safetensors::tensor::SafeTensors;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use tch::Tensor;
use glob::glob;
use std::sync::Arc;

pub struct Transformer {
    pub config: Config,
    pub token_embedding_table: Arc<Tensor>,
    pub blocks: Vec<TransformerBlock>,
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

            let norm1_weight = Tensor::ones(&[1, 1, config.hidden_size as i64], (tch::Kind::Float, tch::Device::Cpu));
            let norm2_weight = Tensor::ones(&[1, 1, config.hidden_size as i64], (tch::Kind::Float, tch::Device::Cpu));
            let transformer_block = TransformerBlock::new(
                attention,
                feed_forward,
                norm1_weight,  // norm1 weights (placeholder)
                norm2_weight,  // norm2 weights (placeholder)
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
        let config: Config = Config::load_config(dir)?;

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
        let token_embedding_table = load_tensor(&safetensors, "model.embed_tokens.weight")?
        .to_kind(tch::Kind::Float).into();  // Ensure it's float

        // Load transformer blocks
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::from_safetensors(&safetensors, config.clone(), i)?;
            blocks.push(block);
        }

        Ok(Transformer {
            config,
            token_embedding_table,
            blocks,
        })
    }

pub fn forward(
    &mut self,
    input: &Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    pos: usize,
) -> Result<Tensor, Box<dyn Error + Sync + Send>> {
    // Debugging: Print input tensor shape
    println!("Model forward input shape: {:?}", input.size());
    // assert_eq!(input.size()[1], self.config.max_position_embeddings as i64, "Input sequence length doesn't match max_position_embeddings");

    // Shallow clone the input tensor so we can modify it in-place
    let mut output = input.shallow_clone();

    // Iterate over transformer blocks and update the output tensor
    for (i, block) in self.blocks.iter_mut().enumerate() {
        // Debugging: Print which block is being processed
        println!("Processing Transformer block: {}", i);

        match block.forward(&output, key_cache, value_cache,  pos) {
            Ok(new_output) => {
                output = new_output;
                println!("Transformer output shape after block {}: {:?}", i, output.size());
            },
            Err(e) => {
                println!("Error in block {}: {:?}", i, e);
                return Err(e as Box<dyn Error + Sync + Send>);  // Return the error with Sync + Send
            }
        }
    }

    // Return the final output tensor if all blocks succeed
    Ok(output)
}

   pub fn get_embeddings(&self, tokens: &[usize]) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        // Create a vector to hold the embeddings
        let mut embeddings = vec![0.0; tokens.len() * self.config.hidden_size];

        // Iterate through tokens and extract their corresponding embeddings
        for (i, &token) in tokens.iter().enumerate() {
            // Extract the embedding for the token from the embedding table using `.narrow()`
            let embedding = self.token_embedding_table.get(token as i64);
            
            // Ensure embedding has the correct shape
            assert_eq!(embedding.size()[0] as usize, self.config.hidden_size, "Embedding size mismatch");

            // Copy embedding into the output `embeddings` vector
            embedding.copy_data(
                &mut embeddings[i * self.config.hidden_size..(i + 1) * self.config.hidden_size], 
                self.config.hidden_size
            );
        }

        Ok(embeddings) // Return the embeddings
    }
}
