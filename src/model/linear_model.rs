use super::Model;
use crate::matrix::Matrix;




/// Model which (for response `y`, features `X`, and model parameters `b`) minimizes the objective
/// function (with respect to `b`): `(y - X . b)^T (y - X . b) + lambda * (alpha * |b| + (1-alpha) * b^T . b)`
/// 
/// This is the Elastic Net. Note that for: lambda = 0 this reduces to OLS, for alpha = 1 this reduces to Lasso
/// regression, and for alpha = 0, this reduces to Ridge regression.
pub struct LinearModel<const K_VARS: usize> {
    pub c: [f64 ; K_VARS],
    l1_coeff: f64,
    l2_coeff: f64,
}

impl<const K: usize> LinearModel<K> {}

impl<const K_VARS: usize> Model<K_VARS> for LinearModel<K_VARS> {
    type FittingOptions = Options;

    fn fit<const N: usize>(x: crate::matrix::Matrix<N, K_VARS>, y: crate::matrix::Matrix<N,1>, options: Self::FittingOptions) -> Self {
        if options.lambda == 0.0 {
            // OLS via QR decomposition
            let (q,r) = x.qr();
            let coeff = y.t_dot(&q).dot_t(&Matrix::back_sub_inv(&r));

            return LinearModel {
                c: coeff[0],
                l1_coeff: 0.0,
                l2_coeff: 0.0,
            };

        } else if options.alpha == 1.0 {
            // LASSO Regression via Coordinate Descent
            let mut out = LinearModel {
                c: [0.0 ; K_VARS],
                l1_coeff: options.lambda,
                l2_coeff: 0.0,
            };
            // prep for R^2 calculation
            let mut y_mean = 0.0;
            for i in 0..N {
                y_mean += y[i][0];
            }
            y_mean /= N as f64;

            let mut tss = 0.0;
            for i in 0..N {
                tss += (y[i][0] - y_mean)*(y[i][0] - y_mean);
            }
            // setup of other state variables & constants
            let mut old_rsq = 0.0;
            let mut new_rsq = old_rsq + 1.0 + options.fit_tol;
            let xtx = x.t_dot(&x);

            // coordinate descent
            while f64::abs(old_rsq - new_rsq) > options.fit_tol {
                old_rsq = new_rsq;

                // loop thru coeffs
                for i in 0..K_VARS {
                    out.c[i] = 0.0;
                    let mut new_coeff = 0.0;

                    for j in 0..N {
                        new_coeff += x[j][i] * y[j][0];
                        for k in 0..K_VARS {
                            new_coeff -= x[j][i]*x[j][k]*out.c[k];
                        }
                    }

                    out.c[i] = new_coeff;
                    out.c[i] /= xtx[i][i];

                    // soft-thresholding
                    let thresh = options.lambda / xtx[i][i];
                    out.c[i] = f64::signum(out.c[i]) * f64::max(0.0, f64::abs(out.c[i]) - thresh);
                }

                // compute rsq to check for convergence
                new_rsq = 0.0;
                for i in 0..N {
                    let mut mid = 0.0;
                    for k in 0..K_VARS {
                        mid += x[i][k]*out.c[k];
                    }
                    new_rsq += (y[i][0] - mid)*(y[i][0] - mid);
                }
                new_rsq = 1.0 - new_rsq/tss;
            }

            return out;

        } else if options.alpha == 0.0 {
            // Ridge Regression via QR decomposition
            let mut decomp = x.t_dot(&x);
            for i in 0..K_VARS {
                decomp[i][i] += options.lambda;
            }

            let (q,r) = decomp.qr();
            let c = y.t_dot(&x).dot(&q).dot_t(&Matrix::back_sub_inv(&r))[0];

            return LinearModel {
                c,
                l1_coeff: 0.0,
                l2_coeff: options.lambda,
            };

        } else {
            // Generalized Elastic Net via Coordinate Descent
            let mut out = LinearModel {
                c: [0.0 ; K_VARS],
                l1_coeff: options.lambda * options.alpha,
                l2_coeff: options.lambda * (1.0 - options.alpha),
            };
            // prep for R^2 calculation
            let mut y_mean = y[0][0];
            for i in 1..N {
                y_mean *= i as f64;
                y_mean += y[i][0];
                y_mean /= i as f64 + 1.0;
            }
            let mut tss = 0.0;
            for i in 0..N {
                tss += (y[i][0] - y_mean)*(y[i][0] - y_mean);
            }
            // setup of other state variables & constants
            let mut old_rsq = 0.0;
            let mut new_rsq = old_rsq + 1.0 + options.fit_tol;
            let xtx = x.t_dot(&x);

            // coordinate descent
            while f64::abs(old_rsq - new_rsq) > options.fit_tol {
                old_rsq = new_rsq;

                // loop thru coeffs
                for i in 0..K_VARS {
                    out.c[i] = 0.0;
                    let mut new_coeff = 0.0;

                    for j in 0..N {
                        new_coeff += x[j][i] * y[j][0];
                        for k in 0..K_VARS {
                            new_coeff -= x[j][i]*x[j][k]*out.c[k];
                        }
                    }

                    out.c[i] = new_coeff;
                    out.c[i] /= xtx[i][i];

                    // soft-thresholding
                    let thresh = options.lambda * options.alpha / xtx[i][i];
                    out.c[i] = f64::signum(out.c[i]) * f64::max(0.0, f64::abs(out.c[i]) - thresh);

                    // shrinkage
                    out.c[i] /= 1.0 + options.lambda*(1.0 - options.alpha);
                }

                // compute rsq to check for convergence
                new_rsq = 0.0;
                for i in 0..N {
                    let mut mid = 0.0;
                    for k in 0..K_VARS {
                        mid += x[i][k]*out.c[k];
                    }
                    new_rsq += (y[i][0] - mid)*(y[i][0] - mid);
                }
                new_rsq = 1.0 - new_rsq/tss;
            }

            return out;
        }
    }

    fn predict<const N: usize>(&self, x: &Matrix<N, K_VARS>) -> Matrix<N,1> {
        x.dot_t(&Matrix::new([self.c]))
    }
}


pub struct Options {
    lambda: f64,
    alpha: f64,
    fit_tol: f64, // threshold for R^2 convergence used in coordinate descent for fitting the LASSO and Elastic Net
}




#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::model::Model;

    use super::{LinearModel, Options};
    
    const SEED: u64 = 69;
    const TOL: f64 = 0.1;

    #[test]
    pub fn test_ols() {
        let x: Matrix<10000, 4>=Matrix::rand_seeded(SEED);
        let true_coeff = Matrix::new([[1.0],[2.0],[3.0],[4.0]]);
        let y = x.dot(&true_coeff);

        let options = Options {
            lambda: 0.0,
            alpha: 0.0,
            fit_tol: 0.0, // free param
        };
        let model = LinearModel::fit(x, y, options);
        println!("{:?}", model.c);
        for i in 0..4 {
            assert!(f64::abs(model.c[i] - true_coeff[i][0]) < TOL);
        }

    }
    
    #[test]
    pub fn test_ridge() {
        let x: Matrix<10000, 4>=Matrix::rand_seeded(SEED);
        let true_coeff = Matrix::new([[1.0],[2.0],[3.0],[4.0]]);
        let y = x.dot(&true_coeff);

        let options = Options {
            lambda: 10.0,
            alpha: 0.0,
            fit_tol: 0.0, // free param
        };
        let model = LinearModel::fit(x, y, options);
        println!("{:?}", model.c);
        for i in 0..4 {
            assert!(f64::abs(model.c[i] - true_coeff[i][0]) < TOL);
        }
    }
    #[test]
    pub fn test_lasso() {
        let x: Matrix<10000, 4>=Matrix::rand_seeded(SEED);
        let true_coeff = Matrix::new([[1.0],[2.0],[3.0],[4.0]]);
        let y = x.dot(&true_coeff);

        let options = Options {
            lambda: 0.01,
            alpha: 1.0,
            fit_tol: 1e-16,
        };
        let model = LinearModel::fit(x, y, options);
        println!("{:?}", model.c);
        for i in 0..4 {
            assert!(f64::abs(model.c[i] - true_coeff[i][0]) < TOL);
        }
    }
    #[test]
    pub fn test_elastic() {
        let x: Matrix<10000, 4>=Matrix::rand_seeded(SEED);
        let true_coeff = Matrix::new([[1.0],[2.0],[3.0],[4.0]]);
        let y = x.dot(&true_coeff);

        let options = Options {
            lambda: 0.01,
            alpha: 0.99,
            fit_tol: 1e-16,
        };
        let model = LinearModel::fit(x, y, options);
        println!("{:?}", model.c);
        for i in 0..4 {
            assert!(f64::abs(model.c[i] - true_coeff[i][0]) < TOL);
        }
    }
}