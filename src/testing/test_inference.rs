// use tch::Tensor;
// use std::error::Error;
// use crate::transformers::transformer::Transformer;
// use crate::tokenizer::hf_tokenizer::ModelTokenizer;
// use crate::sampler::Sampler;

// pub fn test_inference(
//     model: &mut Transformer, 
//     tokenizer: &ModelTokenizer,   // Use your tokenizer here
//     sampler: &mut Sampler,       // Use your sampler here
//     input_text: &str         // Input as text
// ) -> Result<String, Box<dyn Error + Send + Sync>> {
//     // Step 1: Tokenize the input text
//     let input_token_ids = match tokenizer.encode(input_text) {
//     Ok(ids) => ids,
//     Err(e) => {
//         println!("Error encoding text: {:?}", e);
//         return Err(e);  // or handle it another way
//     }
// };


//     // Convert token IDs to a tensor
//     // Convert token IDs to a tensor
//     let input_token_ids_i64: Vec<i64> = input_token_ids.iter().map(|&id| id as i64).collect();  // Convert Vec<u32> to Vec<i64>
//     let input_tensor = Tensor::f_from_slice(&input_token_ids_i64)?.to_kind(tch::Kind::Int64).unsqueeze(0);  // [1, seq_len]



//     // Step 2: Initialize key and value caches (dummy example sizes)
//     let num_layers = model.config.num_hidden_layers * 2;
//     let head_dim = model.config.get_head_dim();
//     let seq_len = input_token_ids.len();

//     let mut key_cache: Vec<f32> = vec![0.0; num_layers * head_dim * seq_len];   
//     let mut value_cache: Vec<f32> = vec![0.0; num_layers * head_dim * seq_len];

//     // Step 3: Perform inference
//     let pos = 0;  // Start position for attention (could be 0 for a simple case)
//     let output = model.forward(&input_tensor, &mut key_cache, &mut value_cache, seq_len, pos);

//     // Step 4: Sample from the output
//    let output_floats: Vec<f32> = Vec::<f32>::try_from(output)?;  // Convert Tensor to Vec<f32>
//    let sampled_token_id = sampler.sample(&output_floats);    // Pass slice &[f32] to sampler

//     // Step 5: Decode the sampled token back to text
//    // Pass a slice of Vec<u32> to tokenizer.decode
//    let generated_text = tokenizer.decode(&vec![sampled_token_id])?;


//     Ok(generated_text)  // Return the generated text
// }