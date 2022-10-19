# `auto_ml`: universal regressor
This project aims to automate the science of prediction. Feature engineering, model fitting/selection, and productionization are all handled by this package.

## Feature Engineering 
This portion of the project aims to take data whose structure has been specified and develop a large number of features on the set. This module can easily be switched out of the pipeline, and you can feed the fitting/selection module pre-featured data. 

This module with be built last.

Planned Feature Analyses:
1. Point-to-point statistics
2. Rolling statistics, EWMAs, lags, etc.
3. Spectral Analysis


## Model Fitting and Selection
The pipeline produces a "model graph" of model components (e.g. convolutions, trees, regressions), with an associated list of hyperparameters. A search over hyperparameter space is performed using out-of-sample data and the best model is selected based on various scoring metrics.

## Productionization
The models can be easily saved and loaded for extensive production use.