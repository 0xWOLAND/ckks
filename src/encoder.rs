use ndarray::{Array1, Array2};
use ndarray_linalg::{c64, error::LinalgError, Scalar, Solve};

use crate::plaintext::Plaintext;

struct CkksEncoder {
    m: usize,
    unity: c64,
}

impl CkksEncoder {
    pub fn new(m: usize) -> CkksEncoder {
        let unity = (2f64 * std::f64::consts::PI * c64::i()).exp();
        CkksEncoder { m, unity }
    }

    pub fn vandermonde(m: usize, xi: c64) -> Array2<c64> {
        let n = m / 2;
        let mut mat = vec![];

        for i in 0..n {
            let root = xi.powu(2 * i as u32 + 1);
            for j in 0..n {
                mat.push(root.powu(j as u32));
            }
        }

        let res: Array2<c64> = Array2::from_shape_vec((n, n), mat).unwrap();
        res
    }

    fn sigma_inverse(self, b: Array1<c64>) -> Result<Plaintext, LinalgError> {
        let a = CkksEncoder::vandermonde(self.m, self.unity);
        let coeffs = a.solve_into(b)?;
        Ok(Plaintext::new(coeffs))
    }
}
