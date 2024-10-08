use std::error::Error;
use tch::Tensor;
use crate::utils::{gelu, matmul};
use tch::IndexOp;
use safetensors::SafeTensors;
use std::sync::Arc;

pub struct FeedForward {
    pub w1: Arc<Tensor>,
    pub w2: Arc<Tensor>,
    pub w3: Arc<Tensor>,
    pub dim: usize,
    pub hidden_dim: usize,
}

impl FeedForward {
    pub fn new(
        w1: Arc<Tensor>,
        w2: Arc<Tensor>,
        w3: Arc<Tensor>,
        dim: usize,
        hidden_dim: usize,
    ) -> Self {
        FeedForward {
            w1,
            w2,
            w3,
            dim,
            hidden_dim,
        }
    }

    pub fn forward(&self, input: Arc<Tensor>) -> Result<Arc<Tensor>, Box<dyn Error>> {
        // Manual matrix multiplication for the two projections
        let gate_output = matmul(&input, &self.w1)?;  // matmul returns Tensor
        let projection_output = matmul(&input, &self.w3)?;  // matmul returns Tensor

        // Apply activation (GELU) and element-wise multiplication
        let intermediate = Tensor::zeros(gate_output.size(), (tch::Kind::Float, gate_output.device()));
        for i in 0..gate_output.size()[0] {
            for j in 0..gate_output.size()[1] {
                let x: f32 = gate_output.double_value(&[i as i64, j as i64]) as f32;
                let activated = gelu(x);
                let proj = projection_output.double_value(&[i as i64, j as i64]) as f32;
                intermediate.i((i as i64, j as i64)).fill_((activated * proj) as f64);
            }
        }

        // Perform the final linear transformation
        let final_output = matmul(&intermediate, &self.w2)?;  // matmul returns Tensor

        Ok(Arc::new(final_output))  // Wrap final_output in Arc<Tensor>
    }

    pub fn from_safetensors(
        st: &SafeTensors,
        layer_index: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // these are not Tensor methods. Needs its own method.
        let w1 = Arc::new(Tensor::from_safetensors(st, format!("w1_{}", layer_index))?);
        let w2 = Arc::new(Tensor::from_safetensors(st, format!("w2_{}", layer_index))?);
        let w3 = Arc::new(Tensor::from_safetensors(st, format!("w3_{}", layer_index))?);
        
        // Derive dim and hidden_dim from tensor shapes
        let dim = w1.size()[1] as usize;
        let hidden_dim = w1.size()[0] as usize;

        Ok(FeedForward {
            w1,
            w2,
            w3,
            dim,
            hidden_dim,
        })
    }
}
