use tch::{Tensor, Kind};
use std::error::Error;
use safetensors::tensor::TensorView;


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

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error>> {
    let a_dims = a.size();
    let b_dims = b.size();
    
    if a_dims[1] != b_dims[0] {
        return Err("Invalid dimensions for matrix multiplication".into());
    }

    let mut result = Tensor::zeros(&[a_dims[0], b_dims[1]], (tch::Kind::Float, a.device()));

    for i in 0..a_dims[0] {
        for j in 0..b_dims[1] {
            let mut sum = 0.0;
            for k in 0..a_dims[1] {
                sum += (a.double_value(&[i, k]) * b.double_value(&[k, j])) as f32;
            }
            result.get(i).get(j).fill_(sum);
        }
    }

    Ok(result)
}

pub fn gelu(x: f32) -> f32 {
    // Approximation of GELU
    0.5 * x * (1.0 + f32::tanh(f32::sqrt(2.0 / std::f32::consts::PI) * (x + 0.044715 * x.powi(3))))
}