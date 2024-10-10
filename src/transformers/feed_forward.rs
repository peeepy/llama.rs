use std::error::Error;
use tch::Tensor;
// use crate::utils::{gelu, matmul};
use std::sync::Arc;
// use tch::IndexOp;

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

   pub fn forward(&self, input: Arc<Tensor>) -> Result<Arc<Tensor>, Box<dyn Error + Sync + Send>> {
    // Debugging: Print input shape
    println!("FeedForward input shape: {:?}", input.size());
    println!("FeedForward w1 shape: {:?}", self.w1.size());
    println!("FeedForward w3 shape: {:?}", self.w3.size());

    let batch_size = input.size()[0];
    let seq_len = input.size()[1];
    
    // // Reshape input to [batch_size * seq_len, hidden_dim]
    // let reshaped_input = input.view([-1, self.dim as i64]);
    // println!("Reshaped input shape: {:?}", reshaped_input.size());

    // Compute gate and up-projection
    let gate_output = input.matmul(&self.w1.transpose(0, 1));
    let projection_output = input.matmul(&self.w3.transpose(0, 1));

    println!("FeedForward gate_output shape: {:?}", gate_output.size());
    println!("FeedForward projection_output shape: {:?}", projection_output.size());

    // Apply activation (GELU) and element-wise multiplication
    let intermediate = gate_output.gelu("tanh") * projection_output;
    println!("Intermediate tensor shape after GELU application: {:?}", intermediate.size());

    // Perform the final linear transformation
    let output = intermediate.matmul(&self.w2.transpose(0, 1));
    // println!("Output shape before reshaping: {:?}", output.size());

    // // Reshape output back to [batch_size, seq_len, hidden_dim]
    // let final_output = output.view([batch_size, seq_len, self.dim as i64]);

    // println!("FeedForward final output shape: {:?}", final_output.size());

    Ok(Arc::new(output))
}


}
