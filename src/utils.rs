use tch::Tensor;
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
            Tensor::f_from_slice(&f32_data).to_dtype(Kind::BFloat16, false, false)
        },
        Kind::Int8 => Tensor::f_from_slice(raw_data),
        _ => unreachable!(),
    };

    Ok(tensor.reshape(&shape))
}
// // use crate::quantize_tensors::{ModelTensor, QuantType, TensorData};
// // use std::simd::i32x8;
// // use std::simd::f32x8;

// // // Functions used in NNs
// // pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32, add_unit_offset: bool) {
// //     let n_simd = size/8;

// //     let mut ss_sim = f32x8::ZERO;

// //     for j in 0..n_simd {
// //         let x_vec = f32x8::from(&x[j*8..j*8+8]);
// //         ss_sim += x_vec * x_vec;
// //     }

// //     let mut ss = ss_sim.reduce_add();

// //     ss /= size as f32;
// //     ss += eps;
// //     ss = 1.0 / ss.sqrt();

// //     for j in 0..n_simd {
// //         let x_vec = f32x8::from(&x[j*8..j*8+8]);
// //         let w_vec = f32x8::from(&weight[j*8..j*8+8]);

// //         let r = if add_unit_offset {
// //             ((1.0 + w_vec) * (ss * x_vec)).to_array()
// //         } else {
// //             (w_vec * (ss * x_vec)).to_array()
// //         };

// //         for k in 0..8 {
// //             o[(j*8) + k] = r[k];
// //         }
// //     }
// // }

use tch::Tensor;

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

// // pub fn matmul_q8_0(xout: &mut [f32], x: &[f32], w: &ModelTensor, n: usize) {
// //     assert_eq!(w.quant_type, QuantType::Q8_0);
// //     let w_data = match &w.data {
// //         TensorData::Quantized(data) => data,
// //         _ => panic!("Expected quantized data for Q8_0"),
// //     };

// //     let gs = w.group_size;
// //     let n_simd = gs / 8;

// //     xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
// //         let ni: usize = i * n;

// //         *xout_elem = (0..=(n - gs)).step_by(gs).map(|j| {
// //             let mut ival = i32x8::ZERO;

// //             for k in 0..n_simd {
// //                 let x_vec = i32x8::from_array(x[j+k*8..j+k*8+8].map(|v| v as i32));
// //                 let w_vec = i32x8::from(&w_data[ni+j+k*8..ni+j+k*8+8]);

// //                 ival += x_vec * w_vec;
// //             }

// //             (ival.reduce_add() as f32) * w.scales[(ni + j) / gs]
// //         }).sum()
// //     });
// // }

// // pub fn matmul_q4_0(xout: &mut [f32], x: &[f32], w: &ModelTensor, n: usize) {
// //     assert_eq!(w.quant_type, QuantType::Q4_0);
// //     let w_data = match &w.data {
// //         TensorData::Quantized(data) => data,
// //         _ => panic!("Expected quantized data for Q4_0"),
// //     };

// //     let gs = w.group_size;
// //     let group_size = gs / 2;  // Because Q4_0 packs two values per byte
// //     let n_simd = group_size / 8;

// //     let mask_low = i32x8::splat(0x0F);
// //     let mask_high = i32x8::splat(0xF0);

// //     xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
// //         let ni: usize = i * n / 2;

// //         *xout_elem = (0..=(n/2 - group_size)).step_by(group_size).map(|j| {
// //             let mut ival = i32x8::splat(0);

// //             for k in 0..n_simd {
// //                 let x_vec = i32x8::from_array(x[j*2+k*16..j*2+k*16+16].map(|v| v as i32));
// //                 let w_vec = i32x8::from(&w_data[ni+j+k*8..ni+j+k*8+8]);

// //                 let x_low = x_vec & mask_low;
// //                 let x_high = (x_vec >> 4) & mask_low;

// //                 let w_low = (w_vec & mask_low) - i32x8::splat(8);
// //                 let w_high = ((w_vec & mask_high) >> 4) - i32x8::splat(8);

// //                 ival += x_low * w_low + x_high * w_high;
// //             }

// //             (ival.reduce_add() as f32) * w.scales[(ni + j) / group_size]
// //         }).sum()
// //     });
// // }
