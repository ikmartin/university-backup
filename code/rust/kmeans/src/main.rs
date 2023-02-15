use ndarray::prelude::*;
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand as rand;
use ndarray_rand::rand_distr::Uniform;

fn kmeans(X: &Array2<f64>,k: usize) {
    // initialize the centers
    if 
    let dim = 5;
    let mut centers = vec![Array::random((k,dim), Uniform::new(0., 10.));k];
    println!("{:?}",centers[0])
}

fn main() {
    kmeans(1);
}
