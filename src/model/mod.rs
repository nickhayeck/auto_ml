use crate::matrix::Matrix;

pub mod linear_model;

pub trait Model<const K: usize> {
    type FittingOptions;

    fn fit<const N: usize>(x: Matrix<N, K>, y: Matrix<N, 1>, options: Self::FittingOptions)
        -> Self;
    fn predict<const N: usize>(&self, x: &Matrix<N, K>) -> Matrix<N, 1>;
}
