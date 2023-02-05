use ndarray::prelude::*;
use gamschmidt::mgs;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x).sqrt()
}

fn normalize(mut x: Array1<f64>) -> Array1<f64> {
    let norm = l2_norm(x.view());
    x.mapv_inplace(|e| e/norm);
    x
}

// power method
// returns first column of (a^ta)^{k^2}
fn sing_val(a: Array2::<f64>, k: u32) -> Array2::<f64> {
    // remap a to have f64s
    //a.mapv_inplace(|x| (x as T) as far);

    // store product of a transpose with a
    let mut b = a.t().dot(&a);

    // take 2^k powers of b
    for i in 1..k {
        b = b.dot(&b)
    }
    return b;
}

fn main() {
    // create the matrix
    const NCOLS: usize = 10;
    const NROWS: usize = 10;
    let mut data: [[f64; NCOLS] ; NROWS] = core::array::from_fn::<[f64; NCOLS],NROWS,_>(|i| 
        core::array::from_fn::<f64,NCOLS,_>(|j| if i + j < NCOLS {(i + j + 1) as f64} else {0.} ));

    data.iter().for_each(|row| {
        println!("{:?}", row);
    });

    let a = arr2(&data);
    let b = sing_val(arr2(&data), 6);
    let v1 = b.slice_move(s![..,0]);
    println!("v1: {}", v1);
    println!("Normalized:\n{}",normalize(v1));
}
