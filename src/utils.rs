use tch::{Device, Kind, Tensor};
use std::error::Error;
use safetensors::tensor::TensorView;
use std::sync::Arc;
use std::fs::File;
use std::io::Read;
use safetensors::SafeTensors;
use statrs::function::erf::erf;


// fn get_embeddings(&self, tokens: &[usize]) -> Vec<f32> {
//         let mut embeddings = vec![0.0; tokens.len() * self.config.dim];
//         for (i, &token) in tokens.iter().enumerate() {
//             let embedding_start = token * self.config.dim;
//             let embedding_end = embedding_start + self.config.dim;
//             embeddings[i * self.config.dim..(i + 1) * self.config.dim]
//                 .copy_from_slice(&self.token_embedding_table[embedding_start..embedding_end]);
//         }
//         embeddings
// }

pub fn precomputed_theta_pos_frequencies(head_dim: i64, seq_len: i32, device: &str, theta: f32) -> Tensor {
    // As written in the paper, the dimentions o the embedding must be even
    assert!(head_dim%2==0, "The head_dim must be even");
    // Built the theta parameters
    // According to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,3,..dim/2]
    // Shape: (head_dim / 2)
    let theta_numerator: Tensor = Tensor::range(0, head_dim, (Kind::Float, Device::cuda_if_available()));
    // Shape : (head_dim / 2)
    let theta: &Tensor = 1.0 / (theta ** (theta_numerator / head_dim)).to(device);
    // Construct the positions (the "m" parameter)
    // shape: (seq_len)
    // = torch.arange(seq_len, device=device)
    // multiply each theta by each position using the outer product
    // shape : (seq_len) outer_product * (head_dim / 2) -> (seq_len, head_dim / 2)
    let freq: Tensor = Tensor::outer(m, theta).to(Kind::Float);
    let freq_complex: Tensor = Tensor::polar(Tensor::ones_like(freq), freq);
    // we can computer complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follow
    // shape: (seq_len, head_dim/2) -> (seq-len, head_dim/2)
    return freq_complex
}

pub fn gelu(x: f32) -> f32 {
    // Manually implement the GELU formula
    let erf_part = erf(x as f64 / (2.0f64.sqrt())) as f32;
    0.5 * x * (1.0 + erf_part)
}

pub fn from_tensor_view(view: TensorView) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    let shape: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => Kind::BFloat16,
        safetensors::Dtype::I8 => Kind::Int8,
        _ => return Err("Unsupported dtype for unquantized loading.".into()),
    };

    let raw_data = view.data();
    let tensor = match dtype {
        Kind::BFloat16 => {
            let f32_data: Vec<f32> = bytemuck::cast_slice::<u8, u16>(raw_data)
                .iter()
                .map(|&x| {
                    let bits = x as u32;
                    let sign = bits >> 15;
                    let exp = (bits >> 7) & 0xFF;
                    let mantissa = bits & 0x7F;
                    let f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 16);
                    f32::from_bits(f32_bits)
                })
                .collect();
            Tensor::f_from_slice(&f32_data)?.to_dtype(Kind::BFloat16, false, false)
        },
        Kind::Int8 => Tensor::f_from_slice(raw_data)?,
        _ => unreachable!(),
    };

    Ok(tensor.reshape(&shape))
}

pub fn softmax(x: &mut [f32]){
    let mut sum: f32 = 0.0;
    let mut max_val: f32 = x[0];

    for i in x.iter() {
        if *i > max_val {
            max_val = *i;
        }
    }

    for i in x.iter_mut() {
        *i = (*i - max_val).exp();
        sum += *i;
    }

    for i in x.iter_mut() {
        *i /= sum;
    }
}

pub fn matmul(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>, Box<dyn Error + Sync + Send>> {
    if a.size()[1] != b.size()[0] {
        return Err("Invalid dimensions for matrix multiplication".into());
    }
    
    let result = a.matmul(b);
    
    Ok(Arc::new(result))
}

pub fn list_tensors_in_safetensors(file_path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Open the SafeTensors file
    let mut safetensor_file = File::open(file_path)?;
    let mut buffer = Vec::new();
    safetensor_file.read_to_end(&mut buffer)?;

    // Deserialize the SafeTensors data
    let safetensors = SafeTensors::deserialize(&buffer)?;

    // Iterate through the tensor names
    for (name, tensor) in safetensors.tensors() {
        println!("Tensor Name: {}", name);
        println!("Shape: {:?}", tensor.shape());
    }

    Ok(())
}

// pub fn matmul(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>, Box<dyn Error>> {
//     let a_dims = a.size();
//     let b_dims = b.size();
    
//     if a_dims[1] != b_dims[0] {
//         return Err("Invalid dimensions for matrix multiplication".into());
//     }

//     let mut result = Tensor::zeros(&[a_dims[0], b_dims[1]], (tch::Kind::Float, a.device()));

//     for i in 0..a_dims[0] {
//         for j in 0..b_dims[1] {
//             let mut sum = 0.0;
//             for k in 0..a_dims[1] {
//                 sum += (a.double_value(&[i, k]) * b.double_value(&[k, j])) as f32;
//             }
//             // Use index_put_ to directly set the value at (i, j)
//             result.index_put_(&[i.into(), j.into()], &Tensor::from(sum));
//         }
//     }

//     Ok(Arc::new(result))
// }