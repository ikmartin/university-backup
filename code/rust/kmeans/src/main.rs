use ndarray::prelude::*;
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand as nrand;
use ndarray_rand::rand_distr::Uniform;

fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x).sqrt()
}

fn dist2(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let z = x - y;
    return l2_norm(z.view());
}

fn kmeans(X: &Vec<Array1<f64>>, k: usize, iterations: usize) -> usize {
    // initialize the centers
    use rand::{thread_rng};
    let mut rng = thread_rng();
    let mut centers = rand::seq::index::sample(&mut rng, X.len(), k).into_vec();
    println!("The initial centers have indices {:?}", centers);

    // main loop
    for it in 0..iterations {

        // assignment step
        for i in 0..X.len() {
            // find the center which minimizes distance to X[i]
            // eventually update this to user iterators, not for loop
            let mut mindist: f64 = 0.0;
            let mut cent: usize = 0;
            for j in &centers {
                let dist = dist2(&X[i],&X[j]);
                println!("Center index {}, distance {}",j, dist);
                if mindist > dist {
                    mindist = dist;
                    cent = *j;
                }
            }

            println!("Point {} is in center {}", i, cent)
        }
    }

    return 1;
}

fn main() {
    const d: usize = 5;  // ambient dimension of data
    const n: usize = 20; // number of data points
    let k: usize = 5;    // number of clusters
    let it: usize = 50;  // number of iterations

    // construct a vector of data points in R^5 uniformly distributed on 0, 10
    // repeat: repeat Array::random(...) infinitely
    // take: take n of those infinite repetitions
    // collect: stick results of take in a vector
    let data = std::iter::repeat(Array::random(d, Uniform::new(0., 10.))).take(n).collect::<Vec<_>>();

    // run means
    let clusters = kmeans(&data, k, it);
}
