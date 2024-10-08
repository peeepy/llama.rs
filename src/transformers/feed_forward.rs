use crate::tensor::{ModelTensor, QuantType};
use std::error::Error;
use tch::Tensor;
use crate::utils::{gelu, matmul};
use tch::IndexOp;
use safetensors::SafeTensors;

pub struct FeedForward {
    pub w1: ModelTensor,
    pub w2: ModelTensor,
    pub w3: ModelTensor,
    pub quant_type: QuantType,
    pub group_size: usize,
    pub dim: usize,
    pub hidden_dim: usize,
}

impl FeedForward {
    pub fn new(
        w1: ModelTensor,
        w2: ModelTensor,
        w3: ModelTensor,
        quant_type: QuantType,
        group_size: usize,
        dim: usize,
        hidden_dim: usize,
    ) -> Self {
        FeedForward {
            w1,
            w2,
            w3,
            quant_type,
            group_size,
            dim,
            hidden_dim,
        }
    }

     pub fn forward(&self, input: &ModelTensor) -> Result<Tensor, Box<dyn Error>> {
        // Manual matrix multiplication for the two projections
        let gate_output = matmul(&input.data, &self.w1.data)?;
        let projection_output = matmul(&input.data, &self.w3.data)?;

        // Apply activation (GELU) and element-wise multiplication
        let intermediate: Tensor = Tensor::zeros(gate_output.size(), (tch::Kind::Float, gate_output.device()));
        for i in 0..gate_output.size()[0] {
            for j in 0..gate_output.size()[1] {
                let x = gate_output.double_value(&[i as i64, j as i64]) as f32;
                let activated = gelu(x);
                let proj = projection_output.double_value(&[i as i64, j as i64]) as f32;
                intermediate.i((i as i64, j as i64)).fill_((activated * proj) as f64);
            }
        }

        // Perform the final linear transformation
        let final_output = matmul(&intermediate, &self.w2.data)?;

        Ok(final_output)
    }

    pub fn from_safetensors(
        st: &SafeTensors,
        layer_index: usize,
        quant_type: QuantType,
        group_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let w1 = ModelTensor::from_safetensors(st, format!("w1_{}", layer_index))?;
        let w2 = ModelTensor::from_safetensors(st, format!("w2_{}", layer_index))?;
        let w3 = ModelTensor::from_safetensors(st, format!("w3_{}", layer_index))?;
        
        // Derive dim and hidden_dim from tensor shapes
        let dim = w1.data.size()[1] as usize;
        let hidden_dim = w1.data.size()[0] as usize;

        Ok(FeedForward {
            w1,
            w2,
            w3,
            quant_type,
            group_size,
            dim,
            hidden_dim,
        })
    }
}