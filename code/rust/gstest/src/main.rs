// Import openblas_src or another blas source to have the linker find all symbols.
extern crate openblas_src;

use gramschmidt::{
    GramSchmidt,
    Reorthogonalized,
    Result,
};
use ndarray::arr2;

fn main() -> Result<()> {
    let small_matrix = arr2(
        &[[2.0, 0.5, 0.0, 0.0],
          [0.0, 0.3, 0.0, 0.0],
          [0.0, 1.0, 0.7, 0.0],
          [0.0, 0.0, 0.0, 3.0]]
    );
    let mut cgs2 = Reorthogonalized::from_matrix(&small_matrix)?;
    cgs2.compute(&small_matrix)?;
    assert!(small_matrix.all_close(&cgs2.q().dot(cgs2.r()), 1e-14));
    Ok(())
}
