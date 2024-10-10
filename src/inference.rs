use tch::Tensor;
use std::error::Error;
use crate::transformers::transformer::Transformer;
use crate::tokenizer::hf_tokenizer::ModelTokenizer;
use crate::sampler::Sampler;
use crate::utils::softmax;

pub fn simple_inference(
    model: &mut Transformer, 
    tokenizer: &ModelTokenizer,
    sampler: &mut Sampler,
    input_text: &str
) -> Result<String, Box<dyn Error + Send + Sync>> {
    // Step 1: Tokenize the input text
    let input_token_ids = tokenizer.encode(input_text)?;
    println!("Input text: '{}'", input_text);
    let token_ids: Vec<usize> = input_token_ids.iter().map(|&id| id as usize).collect();
    println!("Number of tokens after encoding: {}", token_ids.len());

    // Step 2: Get embeddings for the token IDs
    let embeddings_vec = model.get_embeddings(&token_ids)?;
    println!("Embeddings vector length: {}", embeddings_vec.len());

    // Convert the embeddings to a tensor
    let mut input_tensor = Tensor::from_slice(&embeddings_vec)
        .view([1, token_ids.len() as i64, model.config.hidden_size as i64]);
    println!("Initial input tensor shape: {:?}", input_tensor.size());

    // Handle variable-length inputs
    let max_seq_len = 8192; // model.config.max_position_embeddings; not using this bc rope
    println!("max_seq_len: {}", max_seq_len);
    let batch_size = 1;
    let seq_len = token_ids.len();
    println!("seq_len: {}", seq_len);
    let hidden_dim = model.config.hidden_size;
    println!("hidden_dim: {}", hidden_dim);

    if seq_len < max_seq_len {
        // Pad the input
        let padding = Tensor::zeros(&[batch_size, (max_seq_len - seq_len) as i64, hidden_dim as i64], (input_tensor.kind(), input_tensor.device()));
        input_tensor = Tensor::cat(&[input_tensor, padding], 1);
    } else if seq_len > max_seq_len {
        // Truncate the input
        input_tensor = input_tensor.narrow(1, 0, max_seq_len as i64);
    }
    println!("Input tensor shape after padding/truncation: {:?}", input_tensor.size());
    assert_eq!(input_tensor.size()[1], max_seq_len as i64, "Sequence length mismatch after padding/truncation");

    // Step 3: Initialize key and value caches
    let num_layers = model.config.num_hidden_layers;
    let head_dim = model.config.get_head_dim();
    let num_heads = model.config.num_attention_heads;
    let mut key_cache = Tensor::zeros(
    &[batch_size, num_heads as i64, max_seq_len as i64, head_dim as i64],
    (tch::Kind::Float,input_tensor.device()),
    );

    let mut value_cache = Tensor::zeros(
    &[batch_size, num_heads as i64, max_seq_len as i64, head_dim as i64],
    (tch::Kind::Float,input_tensor.device()),
    );

    // Step 4: Perform inference
    let pos = seq_len - 1;  // Position for attention (last token of input)
    let output = model.forward(&input_tensor, &mut key_cache, &mut value_cache, pos)?;
    println!("Model output shape: {:?}", output.size());

    // Step 5: Sample from the output
    let last_token_logits = output.select(1, -1); // Select the last sequence position
    println!("Last token logits shape: {:?}", last_token_logits.size());

    // Convert to f32
    let mut output_floats: Vec<f32> = Vec::<f32>::try_from(last_token_logits.view(-1))?;
    println!("Output floats length: {}", output_floats.len());
    println!("First few logits: {:?}", &output_floats.iter().take(5).collect::<Vec<_>>());

    // Apply softmax to convert logits to probabilities
    softmax(&mut output_floats);
    println!("First few probabilities: {:?}", &output_floats.iter().take(5).collect::<Vec<_>>());

    let sampled_token_id = sampler.sample(&output_floats);
    println!("Sampled token ID: {}", sampled_token_id);

    // Step 6: Decode the sampled token back to text
    let generated_text = tokenizer.decode(&vec![sampled_token_id])?;
    println!("Generated text: {:?}", generated_text);

    // Optional: Print the top K most likely tokens
    let k = 5;  // Change this to see more or fewer top tokens
    let mut indexed_probs: Vec<(usize, f32)> = output_floats.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top {} most likely tokens:", k);
    for (idx, prob) in indexed_probs.iter().take(k) {
        if let Ok(token) = tokenizer.decode(&vec![*idx as u32]) {
            println!("Token ID: {}, Probability: {:.4}, Text: {:?}", idx, prob, token);
        }
    }

    Ok(generated_text)  // Return the generated text
}

