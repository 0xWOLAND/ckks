use ndarray::{Array, Array1};
use ndarray_linalg::c64;
use std::ops::*;

pub struct Plaintext(pub Array1<c64>);

impl Plaintext {
    pub fn new(v: Array1<c64>) -> Plaintext {
        Self(v)
    }

    // eval
    pub fn eval(&self, root: c64) -> c64 {
        self.0
            .iter()
            .enumerate()
            .fold(c64::new(0f64, 0f64), |sum, (i, &x)| {
                sum + root.powu(i as u32) * x
            })
    }
    // magnitude
    pub fn mag(&self) -> c64 {
        self.0
            .iter()
            .fold(c64::new(0f64, 0f64), |sum, &x| sum + x.powi(2))
            .powf(0.5)
    }
}

impl Add for Plaintext {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Plaintext(self.0 + rhs.0)
    }
}

impl Mul for Plaintext {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let d = self.0.len() + rhs.0.len() - 1;
        let mut poly = Array::zeros(d);
        for i in 0..self.0.len() {
            for j in 0..rhs.0.len() {
                poly[i + j] += self.0[i] * rhs.0[j];
            }
        }
        Plaintext(poly)
    }
}

impl Div<usize> for Plaintext {
    type Output = Self;

    fn div(self, rhs: usize) -> Self {
        Plaintext(self.0.mapv(|x| x / rhs as f64))
    }
}
