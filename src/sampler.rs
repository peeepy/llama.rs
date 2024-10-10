use crate::utils;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Copy, Clone)]
struct ProbIndex {
    prob: f32,
    index: u32,
}

pub struct Sampler {
    vocab_size: u32,
    probindex: Vec<ProbIndex>,
    temperature: f32,
    top_p: f32,
    seed: ChaCha8Rng,
}

impl Sampler {
    pub fn new(vocab_size: u32, temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            vocab_size,
            probindex: vec![
                ProbIndex {
                    prob: 0.0,
                    index: 0
                };
                vocab_size as usize
            ],
            temperature,
            top_p,
            seed: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            panic!("Empty logits array");
        }

        let mut working_logits = logits.to_vec();

        if self.temperature == 0.0 {
            return self.sample_argmax(&working_logits);
        }

        self.apply_temperature(&mut working_logits);
        utils::softmax(&mut working_logits);

        let rand = self.get_random_float();

        if self.top_p <= 0.0 || self.top_p >= 1.0 {
            self.sample_multinomial(&working_logits, rand)
        } else {
            self.sample_top_p(&working_logits, rand)
        }
    }

    fn sample_argmax(&self, probabilities: &[f32]) -> u32 {
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index as u32)
            .unwrap_or(0)
    }

    fn sample_multinomial(&self, probabilities: &[f32], rand: f32) -> u32 {
        probabilities
            .iter()
            .scan(0.0, |sum, &p| {
                *sum += p;
                Some(*sum)
            })
            .position(|cdf| rand < cdf)
            .map(|i| i as u32)
            .unwrap_or((probabilities.len() - 1) as u32)
    }

    fn sample_top_p(&mut self, probabilities: &[f32], rand: f32) -> u32 {
        self.prepare_top_p(probabilities);
        let cutoff = self.find_top_p_cutoff();
        self.sample_from_top_p(rand, cutoff)
    }

    fn prepare_top_p(&mut self, probabilities: &[f32]) {
        self.probindex.clear();
        self.probindex
            .extend(probabilities.iter().enumerate().map(|(i, &p)| ProbIndex {
                prob: p,
                index: i as u32,
            }));
        self.probindex
            .sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    }

    fn find_top_p_cutoff(&self) -> usize {
        let mut cumulative_prob = 0.0;
        self.probindex
            .iter()
            .take_while(|pi| {
                cumulative_prob += pi.prob;
                cumulative_prob <= self.top_p
            })
            .count()
    }

    pub fn sample_from_top_p(&self, rand: f32, cutoff: usize) -> u32 {
        if cutoff == 0 || self.probindex.is_empty() {
            return 0; // Return a default value if cutoff is 0 or probindex is empty
        }
        
        let safe_cutoff = cutoff.min(self.probindex.len());
        let total_prob: f32 = self.probindex[..safe_cutoff].iter().map(|pi| pi.prob).sum();
        
        if total_prob <= 0.0 {
            return self.probindex[0].index; // Return the most probable token if total_prob is not positive
        }
        
        let scaled_rand = rand * total_prob;
        let mut cumulative_prob = 0.0;
        for pi in &self.probindex[..safe_cutoff] {
            cumulative_prob += pi.prob;
            if scaled_rand < cumulative_prob {
                return pi.index;
            }
        }
        self.probindex[safe_cutoff - 1].index
    }

    fn apply_temperature(&self, logits: &mut [f32]) {
        logits.iter_mut().for_each(|l| *l /= self.temperature);
    }

    fn get_random_float(&mut self) -> f32 {
        self.seed.gen()
    }
}
