// rmsnorm.rs
pub fn rmsnorm(output: &mut [f32], input: &[f32], weight: &[f32], size: usize, epsilon: f32) {
    // Compute the RMS of the input
    let mut rms = 0.0;
    for i in 0..size {
        rms += input[i] * input[i];
    }
    rms = (rms / size as f32).sqrt();

    // Normalize the input and apply the weight
    let scale = 1.0 / (rms + epsilon);
    for i in 0..size {
        output[i] = input[i] * scale * weight[i];
    }
}
