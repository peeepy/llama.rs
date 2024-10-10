use std::error::Error;
use tch::Tensor;

pub fn rmsnorm(
    input: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<Tensor, Box<dyn Error + Sync + Send>> {
    // Compute the mean square along the last dimension
    let mean_square = input.square().mean_dim(&[-1i64][..], true, tch::Kind::Float);

    // Compute RMS
    let rms = (mean_square + epsilon as f64).sqrt();

    // Normalize input
    let normed_input = input / rms;

    // Apply weight
    let output = normed_input * weight;

    Ok(output)
}


