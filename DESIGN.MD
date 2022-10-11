# `auto_ml` design doc



### TODO:
#### model/matrix.rs
[ ] handle matrix transposition more cleverly
[ ] move the test data into test data files
[ ] make transpose less bad (boolean in struct definition, check the bool in the Index/IndexMut trait implementations)
[ ] for SVD, translate to column-major form (or do the transpose) before computing G-K Bidiagonalization to improve cache miss rate