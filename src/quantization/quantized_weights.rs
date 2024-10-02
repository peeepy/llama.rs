use safetensors::{tensor::TensorInfo, Dtype, SafeTensorError, SafeTensors};
use safetensors::tensor::TensorView;
use serde_json;
use std::error::Error;
use std::collections::HashMap;
use tch::Tensor;
use std::str::FromStr;

// the key steps are: quantize, shift, clamp, and then pack
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QuantType {
    None,
    Q8_0,
    Q4_0,
}

// ensures that unquantized tensors will be unchanged
#[derive(Debug, PartialEq, Clone)]
pub enum TensorData {
    Quantized(Vec<i8>),
    Unquantized(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    pub data: TensorData,
    pub scales: Vec<f32>,
    pub quant_type: QuantType,
    pub group_size: usize,
    pub original_shape: Vec<usize>,
}
impl QuantizedTensor {
    pub fn new(data: &[f32], quant_type: QuantType, group_size: usize) -> Self {
        let original_shape = vec![data.len()];
        match quant_type {
            QuantType::None => Self::no_quantization(data, group_size, original_shape),
            QuantType::Q8_0 => Self::quantize_q8_0(data, group_size, original_shape),
            QuantType::Q4_0 => Self::quantize_q4_0(data, group_size, original_shape),
        }
    }

    fn no_quantization(data: &[f32], group_size: usize, original_shape: Vec<usize>) -> Self {
    Self {
        data: TensorData::Unquantized(data.to_vec()),
        scales: vec![],
        quant_type: QuantType::None,
        group_size,
        original_shape,
    }
}

    fn quantize_q8_0(data: &[f32], group_size: usize, original_shape: Vec<usize>) -> Self {
        let num_groups = (data.len() + group_size - 1) / group_size;
        // Creates a mutable Vec<T> with capacity of data.len()
        let mut quantized_data = Vec::with_capacity(data.len());
        // These are currently empty. Similar to vector.resize() in C++ but done at initialization
        let mut scales = Vec::with_capacity(num_groups);

        for chunk in data.chunks(group_size) {
            let max_abs = chunk.iter()
                .map(|&x| x.abs())
                .filter(|&x| !x.is_nan())  // Filter out NaN values
                .max_by(|a, b| f32::partial_cmp(a, b).unwrap())
                .unwrap_or(1.0);

            let scale = max_abs / 127.0;
            scales.push(scale);

            let quantized_chunk: Vec<i8> = chunk.iter()
                .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
                .collect();
            quantized_data.extend(quantized_chunk);
        }

        Self {
            data: TensorData::Quantized(quantized_data),
            scales,
            quant_type: QuantType::Q8_0,
            group_size,
            original_shape,
        }
    }

    fn quantize_q4_0(data: &[f32], group_size: usize, original_shape: Vec<usize>) -> Self {
        // TODO: Implement 4-bit quantization
        // Remember to pack two 4-bit values into one i8
        let num_groups = (data.len() + group_size -1) / group_size;
        // 
        let mut quantized_data = Vec::with_capacity(data.len() / 2 + data.len() % 2);
        let mut scales = Vec::with_capacity(num_groups);
        
        for chunk in data.chunks(group_size) {
            let max_abs = chunk.iter()
                .map(|&x| x.abs())
                .filter(|&x| !x.is_nan())  // Filter out NaN values
                .max_by(|a, b| f32::partial_cmp(a, b).unwrap())
                .unwrap_or(1.0);


            // Calculate the scale (consider the range [-8, 7] for 4-bit quantization)
            let scale = max_abs / 7.0;
            scales.push(scale);

            // Process two values at a time
            for pair in chunk.chunks(2) {
                // adding 8.0 and clamping to 0.0, 15.0 is more compatible with i8 & straightforward
                let q1 = ((pair[0] / scale).round() + 8.0).clamp(0.0, 15.0);
                let q2 = if pair.len() > 1 {
                    ((pair[1] / scale).round() + 8.0).clamp(0.0, 15.0)
                } else {
                    0.0 // If q2 is missing, set to 0.0 as the default value
                };

                // Packs q1 and q2 into a single i8 and adds to quantized_data 
                let packed_q4 = (q1 as u8) | ((q2 as u8) << 4);
                quantized_data.push(packed_q4 as i8);
            }
        }

        Self {
        data: TensorData::Quantized(quantized_data),
        scales,
        quant_type: QuantType::Q4_0,
        group_size,
        original_shape,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
    match &self.data {
        TensorData::Unquantized(data) => data.clone(),
        TensorData::Quantized(data) => match self.quant_type {
            QuantType::None => unreachable!(), // This case should never happen
            QuantType::Q8_0 => self.dequantize_q8_0(data),
            QuantType::Q4_0 => self.dequantize_q4_0(data),
        },
    }
}

    fn dequantize_q8_0(&self, data: &[i8]) -> Vec<f32> {
    let mut dequantized = Vec::with_capacity(data.len());
    for (i, &q) in data.iter().enumerate() {
        let scale = self.scales[i / self.group_size];
        dequantized.push(q as f32 * scale);
    }
    dequantized
}

    fn dequantize_q4_0(&self, data: &[i8]) -> Vec<f32> {
        /*
        Start by modifying your loop to handle two values for each i8.
        You might want to use bitwise operations to extract the lower and upper 4 bits from each i8.
        For each extracted 4-bit value, apply the inverse of the operations you did during quantization
        (unshift, unclamp, scale).
        Remember, the goal is to reverse the steps you took during quantization. */
        let data = match &self.data {
        TensorData::Quantized(data) => data, // Use the Quantized vector
        TensorData::Unquantized(_) => {
            panic!("Expected quantized data, found unquantized data.");
        }
    };

        let mut dequantized = Vec::with_capacity(data.len() * 2);
        
        for (i, &packed) in data.iter().enumerate() {
            let scale = self.scales[i / (self.group_size / 2)]; // gs adjusted for 4bit
            // it's packed is one i8. We're adding 0x0F (15) to extract the lower & upper 4 bits
            let unpacked_q1 = (packed & 0x0F) as i8; // upper 4 bits converted to separate i8
            let unpacked_q2 = ((packed >> 4) & 0x0F) as i8; // lower 4 bits converted to separate i8

            let dequantized_q1 = ((unpacked_q1 as f32) - 8.0) * scale; // unshifting from 0,15.0 to -8,7
            let dequantized_q2 = ((unpacked_q2 as f32) - 8.0) * scale; // unshifting from 0,15.0 to -8,7

            dequantized.push(dequantized_q1);
            dequantized.push(dequantized_q2);
        }
        dequantized
    }

    // // saving and loading as safetensors
    // pub fn to_safetensors(&self) -> SafeTensors {
    //     let mut tensors = HashMap::new();
        
    //     // Store quantized data
    //     let data_tensor = match &self.data {
    //         TensorData::Quantized(data) => Tensor::from_slice(data),
    //         TensorData::Unquantized(data) => Tensor::from_slice(data),
    //     };
    //     tensors.insert("data".to_string(), data_tensor);
        
    //     // Store scales
    //     let scales_tensor = Tensor::from_slice(&self.scales);
    //     tensors.insert("scales".to_string(), scales_tensor);
        
    //     // Store metadata
    //     let metadata = HashMap::from([
    //         ("quant_type".to_string(), format!("{:?}", self.quant_type)),
    //         ("group_size".to_string(), self.group_size.to_string()),
    //         ("original_shape".to_string(), format!("{:?}", self.original_shape)),
    //     ]);
        
    //     SafeTensors::new(tensors, Some(metadata)).unwrap()
    // }

    // pub fn from_safetensors(st: SafeTensors) -> Result<Self, Box<dyn Error>> {
    //     let data = st.tensor("data").unwrap().to_vec::<i8>()?;
    //     let scales = st.tensor("scales").unwrap().to_vec::<f32>()?;
        
    //     let metadata = st.metadata().unwrap();
    //     let quant_type = metadata.get("quant_type").unwrap().parse::<QuantType>()?;
    //     let group_size = metadata.get("group_size").unwrap().parse::<usize>()?;
    //     let original_shape: Vec<usize> = serde_json::from_str(metadata.get("original_shape").unwrap())?;
        
    //     Ok(Self {
    //         data: TensorData::Quantized(data),
    //         scales,
    //         quant_type,
    //         group_size,
    //         original_shape,
    //     })
    // }

    

    // Saving a quantized tensor
    // let qtensor = QuantizedTensor::new(&data, QuantType::Q8_0, 32);
    // qtensor.save_to_file("model_weights.qten")?;

    // // Loading a quantized tensor
    // let loaded_qtensor = QuantizedTensor::load_from_file("model_weights.qten")?;
}