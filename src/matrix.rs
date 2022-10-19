use rand::{SeedableRng, Rng};

#[derive(Debug,Clone,Copy)]
pub struct Matrix<const N: usize, const M: usize> {
    pub rows: usize,
    pub cols: usize,

    values: [[f64;M];N],

}

impl<const N: usize, const M: usize> Matrix<N,M> {
    pub fn new(values: [[f64;M];N]) -> Matrix<N,M> {
        return Matrix {
            rows: N,
            cols: M,
            values,
        }
    }
    pub fn empty() -> Matrix<N,M> {
        return Matrix {
            rows: N,
            cols: M,
            values: [[0.0;M];N],
        }
    }
    pub fn eye() -> Matrix<N,N> {
        let mut out = Matrix::empty();
        for i in 0..N {
            out[i][i] = 1.0;
        }
        return out;
    }
    pub fn rand() -> Matrix<N,M> {
        let mut out = Matrix::empty();
        for i in 0..N {
            for j in 0..M {
                out[i][j] = rand::random();
            }
        }
        return out;
    }
    pub fn rand_seeded(seed: u64) -> Matrix<N,M> {
        let mut random_gen = rand::rngs::StdRng::seed_from_u64(seed);

        let mut out = Matrix::empty();
        for i in 0..N {
            for j in 0..M {
                out[i][j] = random_gen.gen();
            }
        }
        return out;
    }
    
    pub fn dot<const K: usize>(&self, other: &Matrix<M,K>) -> Matrix<N,K> {
        let mut out: Matrix<N,K> = Matrix::empty();
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    out[i][j] += self[i][k]*other[k][j];
                }
            }
        }        
        
        return out;
    }
    pub fn dot_t<const K: usize>(&self, other: &Matrix<K,M>) -> Matrix<N,K> {
        let mut out: Matrix<N,K> = Matrix::empty();
        for i in 0..N {
            for j in 0..K {
                for k in 0..M {
                    out[i][j] += self[i][k]*other[j][k];
                }
            }
        }        
        
        return out;
    }
    pub fn t_dot<const K: usize>(&self, other: &Matrix<N,K>) -> Matrix<M,K> {
        let mut out: Matrix<M,K> = Matrix::empty();
        for i in 0..M {
            for j in 0..K {
                for k in 0..N {
                    out[i][j] += self[k][i]*other[k][j];
                }
            }
        }        
        
        return out;
    }

    pub fn t_dot_t<const K: usize>(&self, other: &Matrix<K,N>) -> Matrix<M,K> {
        let mut out: Matrix<M,K> = Matrix::empty();
        for i in 0..N {
            for j in 0..K {
                for k in 0..M {
                    out[k][j] += self[i][k]*other[j][i];
                }
            }
        }        
        
        return out;
    }

    /// inverts an upper-triangular matrix using Back Substitution
    pub fn back_sub_inv(mat: &Matrix<N,M>) -> Matrix<M,N> {
        assert!(N == M, "Matrix must be square");

        let mut out: Matrix<M,N> = Matrix::empty();
        for col in 0..M {
            out[col][col] = 1.0 / mat[col][col];
            
            // start at diagonal and solve up
            if col > 0 {
                for row in (0..col).rev() {
                    // summation of previous entries times their multiplicand in the dot product
                    for k in (row+1)..col+1 {
                        out[row][col] -= out[k][col] * mat[row][k];
                    }
                    out[row][col] /= mat[row][row];
                }
            }
        }
        return out;
    }
    
    /// Uses the Schwarz-Rutishauser algorithm to compute the decomposition 
    /// `A = QR`, where `Q` is orthogonal and `R` is upper triangular. This algo
    /// has computational complexity `N M^2 + N M + M`
    pub fn qr(&self) -> (Matrix<N,M>, Matrix<M,M>) {
        let mut q: Matrix<N,M> = *self;
        let mut r: Matrix<M,M> = Matrix::empty();

        for k in 0..M {
            // Orthogonalization Step
            for i in 0..k {
                // R[i,k] = Q[:,i] @ Q[:,k]
                for j in 0..N {
                    r[i][k] += q[j][i] * q[j][k];
                }
                // Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
                for j in 0..N {
                    q[j][k] -= r[i][k] * q[j][i];
                }
            }
            // Normalization Step
            for j in 0..N {
                r[k][k] += q[j][k] * q[j][k];
            }
            r[k][k] = f64::sqrt(r[k][k]);
            for j in 0..N {
                q[j][k] /= r[k][k];
            }
        }

        return (q, r);
    }

    pub fn svd(&self) -> (Matrix<N,M>, Matrix<M,M>, Matrix<M,M>) {
        //
        // PHASE 1: Golub-Kahan Bi-Diagonalization
        // https://netlib.org/utk/people/JackDongarra/etemplates/node198.html
        //
        let (q, mut b) = self.qr();
        // init u as empty
        let mut u: Matrix<M,M>= Matrix::empty();
        // init v_0 as unit norm
        let mut v: Matrix<M,M> = Matrix::empty();
        v[0][0] = 1.0;

        let mut alpha: f64 = 0.0;
        let mut beta: f64 = 0.0;
        // initial step for k=0!
        for row in 0..M {
            for inner in 0..M {
                u[row][0] += b[row][inner]*v[inner][0];
            }
            alpha += u[row][0]*u[row][0];
        }
        alpha = f64::sqrt(alpha);
        for row in 0..M {
            u[row][0] /= alpha;
        }
        
        v[0][1] -= alpha; // since v_0 = [1 0 0 0 ...] we can do this
        for row in 0..M {
            for inner in 0..M {
                v[row][1] += b[inner][row]*u[inner][0];
            }
            beta += v[row][1]*v[row][1];
        }
        beta = f64::sqrt(beta);
        for row in 0..M {
            v[row][1] /= beta;
        }



        for k in 1..(M-1) {
            alpha = 0.0;
            // u_k  = A . v_k - \beta * u_{k-1}
            for row in 0..M {
                for inner in 0..M {
                    u[row][k] += b[row][inner]*v[inner][k];
                }
                u[row][k] -= beta * u[row][k-1];
                alpha += u[row][k]*u[row][k]
            }
            alpha = f64::sqrt(alpha);
            // u_k  = u_k / \alpha
            for row in 0..M {
                u[row][k] /= alpha;
            }

            beta = 0.0;
            // v_{k+1} = A.T . u_k - \alpha * v_k
            for row in 0..M {
                for inner in 0..M {
                    v[row][k+1] += b[inner][row]*u[inner][k];
                }
                v[row][k+1] -= alpha * v[row][k];
                beta += v[row][k+1]*v[row][k+1]
            }
            beta = f64::sqrt(beta);
            // v_k  = v_k / \beta
            for row in 0..M {
                v[row][k+1] /= beta;
            }
        }

        // last step for u_k
        alpha = 0.0;
        // u_k  = A . v_k - \beta * u_{k-1}
        for row in 0..M {
            for inner in 0..M {
                u[row][M-1] += b[row][inner]*v[inner][M-1];
            }
            u[row][M-1] -= beta * u[row][M-2];
            alpha += u[row][M-1]*u[row][M-1]
        }
        alpha = f64::sqrt(alpha);
        // u_k  = u_k / \alpha
        for row in 0..M {
            u[row][M-1] /= alpha;
        }

        b = u.t_dot(&b).dot(&v);
        let mut u = q.dot(&u);

        // cleanup B
        for i in 0..M {
            for j in 0..M {
                if j != i && j != i+1 {
                    b[i][j] = 0.0;
                }
            }
        }

        //
        // PHASE 2: Demmel-Kahan Bidiagonal SVD
        // http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html
        //
        fn givens(f: f64, g: f64) -> (f64,f64,f64) {
            if f64::abs(f) <= 1e-10 {
                return (0.0,1.0,g);
            } 
            
            if f64::abs(f) > f64::abs(g) {
                let t = g/f;
                let t1 = f64::sqrt(1.0 + t*t);
                return (1.0/t1, t/t1, f*t1)
            } else {
                let t = f/g;
                let t1 = f64::sqrt(1.0 + t*t);
                return (t/t1, 1.0/t1, g*t1)
            }
         }

        let n_iter: usize = 100;
        for _ in 0..n_iter {
            for i in 0..M-1 {
                let mut csr = Matrix::<M,M>::eye();
                let (c,s,_r) = givens(b[i][i], b[i][i+1]);
                csr[i][i] = c;
                csr[i][i+1] = s;
                csr[i+1][i] = -s;
                csr[i+1][i+1] = c;

                b = b.dot_t(&csr);
                v = v.dot_t(&csr);

                let mut csr = Matrix::<M,M>::eye();
                let (c,s,_r) = givens(b[i][i], b[i+1][i]);
                csr[i][i] = c;
                csr[i][i+1] = s;
                csr[i+1][i] = -s;
                csr[i+1][i+1] = c;
                
                b = csr.dot(&b);
                u = u.dot_t(&csr);

            }
        }
        // cleanup b
        for i in 0..M {
            for j in 0..M {
                if j != i {
                    b[i][j] = 0.0;
                }
            }
        }
        (u, b, v)
    }
}

impl<const N: usize, const M: usize> std::ops::Index<usize> for Matrix<N,M> {
    type Output = [f64;M];
    
    fn index(&self, index: usize) -> &Self::Output {
        return self.values.index(index);
    }
}

impl<const N: usize, const M: usize> std::ops::IndexMut<usize> for Matrix<N,M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return self.values.index_mut(index);
    }
}

impl<const N: usize, const M: usize> std::ops::Mul<f64> for Matrix<N,M> {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        for i in 0..N {
            for j in 0..M {
                self[i][j] *= rhs;
            }
        }
        return self;
    }
}

impl<const N: usize, const M: usize> std::ops::Mul<Matrix<N,M>> for f64 {
    type Output = Matrix<N,M>;
    fn mul(self, rhs: Matrix<N,M>) -> Self::Output {
        return rhs*self;
    }
}

impl<const N: usize, const M: usize> std::ops::Neg for Matrix<N,M> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        return -1.0f64*self;
    }
}

impl<const N: usize, const M: usize> PartialEq for Matrix<N,M> {
    fn eq(&self, other: &Self) -> bool {
        for row in 0..N {
            for col in 0..M {
                if self[row][col] != other[row][col] {
                    return false;
                }
            }
        }
        return true;
    }
    fn ne(&self, other: &Self) -> bool {
        return !(self.eq(other))
    }
}
impl<const N: usize, const M: usize> Eq for Matrix<N,M> {}




#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_back_sub_inv() {
        let tol: f64 = 1e-10;

        // Ones
        let mat = Matrix::new([[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0],[0.0,0.0,0.0,1.0]]);
        let id = Matrix::back_sub_inv(&mat).dot(&mat);
        println!("1: {:?}", id);
        
        for row in 0..id.rows {
            for col in 0..id.cols {
                if row == col {
                    assert!(id[row][col]-1.0 < tol);
                } else {
                    assert!(id[row][col] < tol);
                }
            }
        }  

        // Some other values
        let mat = Matrix::new([[1.0,2.0,3.0,4.0],[0.0,10.0,11.0,12.0],[0.0,0.0,25.0,26.0],[0.0,0.0,0.0,90.0]]);
        let id = Matrix::back_sub_inv(&mat).dot(&mat);
        println!("2: {:?}", id);
        
        for row in 0..id.rows {
            for col in 0..id.cols {
                if row == col {
                    assert!(f64::abs(id[row][col]-1.0) < tol);
                } else {
                    assert!(f64::abs(id[row][col]) < tol);
                }
            }
        }
        
        // Large matrix
        let mat = Matrix::new(
            [[402.83014, 967.54968, 538.92053, 960.43089, 848.22684, 612.85705, 192.36716, 510.42718, 62.92912, 74.37880, 717.47705, 440.57557, 148.07402, 9.17194, 51.93606, 675.37240, 433.85055, 408.96311, 96.17596, 71.68482, 764.59300, 50.78826, 904.70776, 234.96827, 324.42137],
            [0.0, 608.28251, 552.42698, 52.75154, 541.22021, 384.77546, 814.96099, 53.58782, 58.30144, 13.42374, 948.15942, 846.46685, 319.06016, 317.62770, 662.59200, 196.26962, 800.74043, 626.02986, 885.59293, 121.74041, 415.78504, 963.04128, 134.14192, 537.10872, 862.84575],
            [0.0, 0.0, 319.24572, 599.89950, 927.19129, 612.31968, 530.51891, 833.23569, 664.40996, 794.21106, 358.12931, 681.35382, 789.69364, 819.99820, 966.47705, 438.87518, 453.73012, 35.75985, 695.28149, 700.94821, 978.65269, 365.78181, 590.17124, 187.66882, 585.56959],
            [0.0, 0.0, 0.0, 696.98126, 295.74022, 761.06019, 79.57783, 190.85517, 272.38186, 684.62500, 721.19703, 451.92311, 163.76083, 479.61627, 902.90911, 137.50433, 944.25781, 975.95233, 927.05014, 667.66401, 959.88454, 499.87795, 116.62374, 408.46331, 141.34027],
            [0.0, 0.0, 0.0, 0.0, 531.35167, 612.88033, 667.38354, 292.17249, 811.22705, 98.06540, 567.44095, 463.48575, 786.49285, 892.24055, 634.32312, 481.86408, 374.50043, 980.65369, 1.95791, 695.79902, 209.27672, 594.70453, 756.69898, 338.45617, 728.24818],
            [0.0, 0.0, 0.0, 0.0, 0.0, 279.57541, 671.47187, 497.51000, 699.52667, 657.53529, 296.54640, 454.08285, 26.65121, 540.60737, 90.87863, 483.57502, 830.79004, 184.95247, 723.61151, 234.77385, 889.87820, 912.75552, 893.63598, 661.29069, 936.90198],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.27106, 548.83096, 325.24698, 805.99686, 488.64418, 210.67814, 10.22804, 165.81546, 773.01493, 39.35039, 508.58457, 437.76625, 507.21554, 6.45559, 91.15966, 413.45300, 138.18446, 38.89145, 590.39408],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 631.93717, 191.82843, 500.54941, 455.28319, 790.03764, 891.38231, 626.03527, 736.56066, 777.94480, 697.67280, 803.72090, 625.29403, 640.16096, 934.55021, 602.13707, 298.58003, 752.99929, 131.96126],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 888.28889, 984.20929, 384.25564, 744.14057, 826.16665, 301.80512, 736.60629, 629.02243, 481.65236, 937.80147, 869.66140, 56.47376, 858.25294, 872.12639, 520.17866, 539.20649, 615.80415],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 871.18611, 235.45834, 742.58912, 888.27830, 534.70055, 235.05005, 65.92619, 710.61201, 98.02908, 173.82306, 447.03558, 101.10453, 511.14283, 445.74860, 527.04317, 755.33585],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 180.40906, 953.19701, 110.96054, 903.73998, 541.50297, 720.16207, 256.36312, 808.26568, 716.60298, 9.16684, 413.64095, 32.25670, 477.39502, 490.31216, 391.34353],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 44.39459, 129.94657, 615.34106, 953.56428, 306.15876, 973.76369, 278.59800, 391.27463, 189.32901, 521.30584, 516.12846, 621.13665, 190.37477, 140.56058],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 531.88571, 945.55883, 491.94951, 267.45152, 861.91029, 357.38698, 929.85030, 341.55126, 292.30271, 325.71555, 177.52443, 615.18918, 704.57264],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 383.34624, 843.50520, 550.49525, 485.95466, 111.84721, 483.31456, 677.05484, 490.53991, 925.19066, 875.45660, 328.27532, 872.47142],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 520.27597, 556.06670, 714.36202, 400.23157, 987.48535, 888.41821, 648.33769, 526.25819, 613.37947, 572.28533, 202.81783],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 470.82414, 513.67908, 394.54760, 646.24029, 682.76681, 270.78483, 132.62086, 569.70310, 819.22950, 864.82491],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 351.24797, 92.00531, 440.08188, 596.08926, 680.61764, 190.48905, 249.44726, 262.20101, 376.79664],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 857.65609, 433.94894, 186.39293, 531.42677, 465.19587, 846.61443, 234.91518, 537.87782],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 277.21736, 218.57641, 632.73815, 595.47425, 884.41407, 506.41143, 631.86264],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 174.03066, 813.26235, 266.19845, 901.87893, 541.48896, 951.89478],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 256.61714, 106.45278, 150.44916, 40.98168, 343.83434],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 122.93510, 564.91154, 904.92466, 123.36255],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 354.89743, 639.10605, 215.73941],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 539.18956, 256.70181],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 362.69526]]
        );
        let id = Matrix::back_sub_inv(&mat).dot(&mat);
        println!("3: {:?}", id);
        
        for row in 0..id.rows {
            for col in 0..id.cols {
                if row == col {
                    assert!(id[row][col]-1.0 < tol);
                } else {
                    assert!(id[row][col] < tol);
                }
            }
        }        
    }


    #[test]
    fn test_qr() {
        let tol = 1e-10;

        let mat = Matrix::new([[1.0,2.0,3.0,4.0],[5.0,10.0,11.0,12.0],[7.0,8.0,25.0,26.0],[11.0,22.0,90.0,90.0]]);
        let (q,r) = mat.qr();

        // check for orthogonality in Q
        let id = q.t_dot(&q);
        for row in 0..id.rows {
            for col in 0..id.cols {
                if row == col {
                    assert!(f64::abs(id[row][col]-1.0) < tol);
                } else {
                    assert!(f64::abs(id[row][col]) < tol);
                }
            }
        }

        // check R is upper-triangular
        for row in 0..r.rows {
            for col in 0..r.cols {
                if row > col {
                    assert!(f64::abs(r[row][col]) < tol);
                }
            }
        }

        // check for equality
        let a = q.dot(&r);
        for row in 0..r.rows {
            for col in 0..r.cols {
                    assert!(f64::abs(a[row][col] - mat[row][col]) < tol);
            }
        }
        
    }

    #[test]
    fn test_ols() {
        let x: [[f64;2];100] = [[0.20309045, 0.10151166],[0.69832198, 0.75820443],[0.01322583, 0.76940789],[0.15427256, 0.58532135],[0.64200034, 0.70975564],[0.80496605, 0.54349374],[0.99747371, 0.55080368],[0.85211516, 0.90730392],[0.85705591, 0.11794594],[0.53353737, 0.66871452],[0.36437744, 0.40937011],[0.78113506, 0.8406983 ],[0.61897722, 0.55977395],[0.83810832, 0.21154069],[0.73656679, 0.1245044 ],[0.082546  , 0.06476062],[0.68742962, 0.60373462],[0.64774728, 0.0174141 ],[0.81642783, 0.24825396],[0.34602423, 0.05429212],[0.99262824, 0.33352599],[0.73623906, 0.43696374],[0.33029814, 0.1506811 ],[0.28106011, 0.07515308],[0.34393494, 0.26979563],[0.19862643, 0.73406526],[0.73830214, 0.27860457],[0.20328576, 0.15232644],[0.63034444, 0.78407657],[0.10460365, 0.99075626],[0.44290315, 0.67208215],[0.9204909 , 0.91431238],[0.53367526, 0.70806906],[0.86631018, 0.19992864],[0.34392655, 0.18261695],[0.04986891, 0.52425228],[0.21565616, 0.50647138],[0.08004812, 0.43791303],[0.35418664, 0.8061326 ],[0.59716041, 0.73127698],[0.62889989, 0.17693399],[0.06200936, 0.43432799],[0.74647617, 0.39112587],[0.76687583, 0.76567987],[0.62231092, 0.24311516],[0.56500536, 0.59170036],[0.31339044, 0.17473874],[0.90245031, 0.94883556],[0.20727813, 0.07547778],[0.92756972, 0.279422  ],[0.33551055, 0.57903914],[0.91566538, 0.18030712],[0.73217712, 0.90289337],[0.1192161 , 0.73607708],[0.50913774, 0.38222208],[0.68050769, 0.73725787],[0.6802044 , 0.62178573],[0.04902013, 0.4271111 ],[0.38744876, 0.98444693],[0.92943455, 0.43321081],[0.4478184 , 0.7623317 ],[0.55186202, 0.67112869],[0.00740711, 0.57869844],[0.96124309, 0.21217936],[0.49878791, 0.76632403],[0.4678897 , 0.81105487],[0.05486548, 0.46812835],[0.48412193, 0.45526029],[0.22127614, 0.65800783],[0.68575901, 0.85960486],[0.24258266, 0.27611659],[0.58066283, 0.60625095],[0.20138329, 0.18299506],[0.31242111, 0.91473176],[0.21362025, 0.64648196],[0.90814031, 0.99393895],[0.13510436, 0.58508398],[0.30873174, 0.32880678],[0.8757608 , 0.3418204 ],[0.44563576, 0.92599586],[0.18352817, 0.45946633],[0.96728224, 0.71977613],[0.91482449, 0.9744755 ],[0.33985651, 0.08793697],[0.56783266, 0.01862503],[0.05210635, 0.51206549],[0.82509386, 0.98115196],[0.10072023, 0.08258031],[0.23396114, 0.44487405],[0.39313069, 0.67959766],[0.02445709, 0.10578177],[0.7464898 , 0.49019164],[0.37419497, 0.28869766],[0.13453407, 0.00706088],[0.50675282, 0.358536  ],[0.04889694, 0.91200545],[0.94855892, 0.25940624],[0.89798132, 0.95755886],[0.73079961, 0.18305238],[0.24117614, 0.53258716]];
        let y: [f64; 100] = [ 2.61696934,  5.71063165,  1.84852054,  2.64913906,  2.74893792, 3.27381423,  4.61966108,  5.70530232,  0.47885984,  1.08360887, 1.89301377,  3.48479983,  3.39109945,  2.19677303,  2.29496562, 0.7530666 ,  3.49143405,  3.46313905, -0.22753867,  2.79309283, 3.56329198,  4.05620408,  1.68275535, -0.50367435,  2.39323642, 3.49841307,  3.48526756,  0.6344203 ,  3.67176673,  3.73416587, 4.43440367,  4.84116357,  2.21031356,  2.33842392,  1.56217862, 1.07525584,  2.44066142,  2.01147707,  4.72900305,  2.73979001, 0.86284485,  0.3356248 ,  2.7821031 ,  2.75341875,  0.82434148, 2.91774651,  0.96189137,  3.79074672,  1.89325587,  2.84055638, 0.57626988,  0.96464406,  3.8999488 ,  3.30463172,  1.6336875 , 4.46051989,  4.17613749,  1.05928008,  4.36607336,  4.63487236, 3.06045699,  4.75311057,  0.19200576,  2.05329817,  2.50241665, 2.41036743,  1.04232854,  0.66022556,  2.03091977,  5.99600616, 1.09858266,  1.25825647, -0.69805932,  4.41929355,  2.97471081, 4.85629811,  1.23983946,  2.76815792,  2.83807811,  4.16464599, 2.55635398,  4.12571333,  4.5070377 ,  0.41648026,  2.12549024, 2.03299705,  3.26748967,  0.13767737,  1.46675777,  2.17537878, 0.1782525 ,  1.43656192,  1.63785763, -0.31367056,  1.67102831, 2.42210074,  2.9994767 ,  5.97269121,  2.18285682,  1.74210526];

        let x = Matrix::new(x);
        let y = Matrix::new([y]);

        let (q,r) = x.qr();

        let beta = Matrix::back_sub_inv(&r).dot_t(&q).dot_t(&y);

        assert!(f64::abs(beta[0][0] - 2.01075233) < 1e-5);
        assert!(f64::abs(beta[1][0] - 3.04392419) < 1e-5);
        
    }

    #[test]
    fn test_svd() {
        let tol = 1e-10;

        let mat = Matrix::new([[1.0,2.0,3.0,4.0],[0.0,10.0,11.0,12.0],[0.0,0.0,25.0,26.0],[0.0,0.0,0.0,90.0]]);
        let (u,s,v) = mat.svd();
        

        // check for orthogonality in U
        let id = u.t_dot(&u);
        for row in 0..id.rows {
            for col in 0..id.cols {
                if row == col {
                    assert!(f64::abs(id[row][col]-1.0) < tol);
                } else {
                    assert!(f64::abs(id[row][col]) < tol);
                }
            }
        }

        // check for orthogonality in V
        let id = v.t_dot(&v);
        for row in 0..id.rows {
            for col in 0..id.cols {
                if row == col {
                    assert!(f64::abs(id[row][col]-1.0) < tol);
                } else {
                    assert!(f64::abs(id[row][col]) < tol);
                }
            }
        }
        
        // check for equality in U.S.Vt == mat
        let id = dbg!(u.dot(&s).dot_t(&v));
        for row in 0..id.rows {
            for col in 0..id.cols {
                assert!(f64::abs(id[row][col] - mat[row][col]) < tol);
            }
        }
        
    }
}
