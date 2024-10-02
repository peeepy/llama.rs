use crate::quantization::quantized_weights::{QuantizedTensor, QuantType, TensorData};
// use std::simd::i32x8;
// use std::simd::f32x8;

// // Functions used in NNs
// pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32, add_unit_offset: bool) {
//     let n_simd = size/8;

//     let mut ss_sim = f32x8::ZERO;

//     for j in 0..n_simd {
//         let x_vec = f32x8::from(&x[j*8..j*8+8]); 
//         ss_sim += x_vec * x_vec;
//     } 

//     let mut ss = ss_sim.reduce_add();

//     ss /= size as f32;
//     ss += eps;
//     ss = 1.0 / ss.sqrt();

//     for j in 0..n_simd {
//         let x_vec = f32x8::from(&x[j*8..j*8+8]);
//         let w_vec = f32x8::from(&weight[j*8..j*8+8]);
        
//         let r = if add_unit_offset {
//             ((1.0 + w_vec) * (ss * x_vec)).to_array()
//         } else {
//             (w_vec * (ss * x_vec)).to_array()
//         };

//         for k in 0..8 {
//             o[(j*8) + k] = r[k];
//         } 
//     } 
// }

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

// pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
//     let n = x.len();
//     let n_simd = n / 8;

//     xout.par_iter_mut().enumerate().for_each(|(i, val)| {
//         let mut sum = f32x8::ZERO;
//         let w_slice = &w[i * n..i * n + n];

//         for j in 0..n_simd {
//             let x_vec = f32x8::from(&x[j*8..j*8+8]);
//             let w_vec = f32x8::from(&w_slice[j*8..j*8+8]);
//             sum += w_vec * x_vec;
//         }

//         *val = sum.reduce_add();
//     });
// }

// pub fn matmul_q8_0(xout: &mut [f32], x: &[f32], w: &QuantizedTensor, n: usize) {
//     assert_eq!(w.quant_type, QuantType::Q8_0);
//     let w_data = match &w.data {
//         TensorData::Quantized(data) => data,
//         _ => panic!("Expected quantized data for Q8_0"),
//     };
    
//     let gs = w.group_size;
//     let n_simd = gs / 8;

//     xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
//         let ni: usize = i * n;

//         *xout_elem = (0..=(n - gs)).step_by(gs).map(|j| {
//             let mut ival = i32x8::ZERO;

//             for k in 0..n_simd {
//                 let x_vec = i32x8::from_array(x[j+k*8..j+k*8+8].map(|v| v as i32));
//                 let w_vec = i32x8::from(&w_data[ni+j+k*8..ni+j+k*8+8]);

//                 ival += x_vec * w_vec;
//             }

//             (ival.reduce_add() as f32) * w.scales[(ni + j) / gs]
//         }).sum()
//     });
// }

// pub fn matmul_q4_0(xout: &mut [f32], x: &[f32], w: &QuantizedTensor, n: usize) {
//     assert_eq!(w.quant_type, QuantType::Q4_0);
//     let w_data = match &w.data {
//         TensorData::Quantized(data) => data,
//         _ => panic!("Expected quantized data for Q4_0"),
//     };
    
//     let gs = w.group_size;
//     let group_size = gs / 2;  // Because Q4_0 packs two values per byte
//     let n_simd = group_size / 8;

//     let mask_low = i32x8::splat(0x0F);
//     let mask_high = i32x8::splat(0xF0);
    
//     xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
//         let ni: usize = i * n / 2;

//         *xout_elem = (0..=(n/2 - group_size)).step_by(group_size).map(|j| {
//             let mut ival = i32x8::splat(0);

//             for k in 0..n_simd {
//                 let x_vec = i32x8::from_array(x[j*2+k*16..j*2+k*16+16].map(|v| v as i32));
//                 let w_vec = i32x8::from(&w_data[ni+j+k*8..ni+j+k*8+8]);

//                 let x_low = x_vec & mask_low;
//                 let x_high = (x_vec >> 4) & mask_low;
                
//                 let w_low = (w_vec & mask_low) - i32x8::splat(8);
//                 let w_high = ((w_vec & mask_high) >> 4) - i32x8::splat(8);

//                 ival += x_low * w_low + x_high * w_high;
//             }

//             (ival.reduce_add() as f32) * w.scales[(ni + j) / group_size]
//         }).sum()
//     });
// }