// the key steps are: quantize, shift, clamp, and then pack
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QuantType {
    None,
    Q8_0,
    Q4_0,
}

// ensures that unquantized tensors will be unchanged
#[derive(Debug, PartialEq, Clone)]
pub enum QuantState {
    Quantized(Vec<i8>),
    Unquantized(Vec<f32>),
}

// TODO: Refactor. Add lifetime arguments, change what is being stored?
#[derive(Debug, PartialEq)]
pub struct ModelTensor {
    pub data: tch::Tensor, // Storing the tensor directly
    pub scales: Vec<f32>,
    pub quant_type: QuantType,
    pub group_size: usize,
    pub original_shape: Vec<usize>,
}

impl ModelTensor {
    // Constructor with quantization options
    pub fn new(tensor: tch::Tensor, quant_type: QuantType, group_size: usize) -> Self {
        let original_shape = tensor
            .size()
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();
        match quant_type {
            QuantType::None => Self::no_quantization(tensor, group_size, original_shape),
            QuantType::Q8_0 => Self::no_quantization(tensor, group_size, original_shape),
            QuantType::Q4_0 => Self::no_quantization(tensor, group_size, original_shape),
        }
    }

    // No quantization: store tensor as-is
    fn no_quantization(tensor: tch::Tensor, group_size: usize, original_shape: Vec<usize>) -> Self {
        Self {
            data: tensor,
            scales: vec![],
            quant_type: QuantType::None,
            group_size,
            original_shape,
        }
    }
}

//     // 8-bit quantization method
//     fn quantize_q8_0(tensor: tch::Tensor, group_size: usize, original_shape: Vec<usize>) -> Self {
//         let flattened_tensor = tensor.view([-1]);
//         let num_groups = (flattened_tensor.size()[0] as usize + group_size - 1) / group_size;
//         let mut scales = Vec::with_capacity(num_groups);
//         let mut quantized_data = vec![];

//         // Loop through groups and quantize
//         for chunk in flattened_tensor.chunk(group_size as i64, 0) {
//             let max_abs = chunk.abs().max();
//             let scale = max_abs / 127.0;
//             scales.push(scale);

//             let quantized_chunk: tch::Tensor = (chunk / scale)
//                 .round()
//                 .clamp(-127.0, 127.0)
//                 .to_kind(tch::Kind::Int8);
//             quantized_data.push(quantized_chunk);
//         }

//         // Convert quantized_data back to a single tensor
//         let quantized_tensor = tch::Tensor::cat(&quantized_data, 0);

//         Self {
//             data: quantized_tensor,
//             scales,
//             quant_type: QuantType::Q8_0,
//             group_size,
//             original_shape,
//         }
//     }

//     // 4-bit quantization method
//     fn quantize_q4_0(tensor: tch::Tensor, group_size: usize, original_shape: Vec<usize>) -> Self {
//         let flattened_tensor = tensor.view([-1]);
//         let num_groups = (flattened_tensor.size()[0] as usize + group_size - 1) / group_size;
//         let mut scales = Vec::with_capacity(num_groups);
//         let mut quantized_data = vec![];

//         // Process two values at a time for 4-bit quantization
//         for chunk in flattened_tensor.chunk(group_size as i64, 0) {
//             let max_abs = chunk.abs().max();
//             let scale = max_abs / 7.0;
//             scales.push(scale);

//             for pair in chunk.chunk(2, 0) {
//                 let q1 = ((pair[0] / scale).round() + 8.0).clamp(0.0, 15.0);
//                 let q2 = if pair.size()[0] > 1 {
//                     ((pair[1] / scale).round() + 8.0).clamp(0.0, 15.0)
//                 } else {
//                     0.0 // Default if missing second value
//                 };

//                 let packed_q4 = (q1 as u8) | ((q2 as u8) << 4);
//                 quantized_data.push(packed_q4 as i8);
//             }
//         }

//         let quantized_tensor = tch::Tensor::of_slice(&quantized_data).to_kind(tch::Kind::Int8);

//         Self {
//             data: quantized_tensor,
//             scales,
//             quant_type: QuantType::Q4_0,
//             group_size,
//             original_shape,
//         }
//     }

//     // Dequantization
//     pub fn dequantize(&self) -> tch::Tensor {
//         match self.quant_type {
//             QuantType::None => self.data.clone(),
//             QuantType::Q8_0 => self.dequantize_q8_0(),
//             QuantType::Q4_0 => self.dequantize_q4_0(),
//         }
//     }

//     // Dequantize Q8_0
//     fn dequantize_q8_0(&self) -> tch::Tensor {
//         let mut dequantized = vec![];
//         let data = self.data.to_kind(tch::Kind::Int8);

//         for (i, &q) in data.iter().enumerate() {
//             let scale = self.scales[i / self.group_size];
//             dequantized.push(q as f32 * scale);
//         }

//         tch::Tensor::f_from_slice(&dequantized).unwrap()
//     }

//     // Dequantize Q4_0
//     fn dequantize_q4_0(&self) -> tch::Tensor {
//         let data = self.data.to_kind(tch::Kind::Int8);
//         let mut dequantized = Vec::with_capacity(data.size()[0] * 2);

//         for (i, &packed) in data.iter().enumerate() {
//             let scale = self.scales[i / (self.group_size / 2)];
//             let unpacked_q1 = (packed & 0x0F) as i8;
//             let unpacked_q2 = ((packed >> 4) & 0x0F) as i8;

//             let dequantized_q1 = ((unpacked_q1 as f32) - 8.0) * scale;
//             let dequantized_q2 = ((unpacked_q2 as f32) - 8.0) * scale;

//             dequantized.push(dequantized_q1);
//             dequantized.push(dequantized_q2);
//         }

//         tch::Tensor::of_slice(&dequantized)
//     }
// }

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
// let qtensor = ModelTensor::new(&data, QuantType::Q8_0, 32);
// qtensor.save_to_file("model_weights.qten")?;

// // Loading a quantized tensor
// let loaded_qtensor = ModelTensor::load_from_file("model_weights.qten")?;
