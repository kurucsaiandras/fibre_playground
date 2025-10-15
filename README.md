# Dense Fibre Composite Generation

## TODO list
(not necessarily in importance order)
 - Implement randomness for fibre diameters
 - Implement statistical descriptors for evaluation
    - For alignment, 3D density distrib. and curvature as well
 - Investigate inhomogenous distribution, touching fibres, clustering
 - Try anisotropic misalignment
 - Try BFGS instead of gradient descent (and compare)
    - Maybe other convergence methods as well
 - Use line segments (or even splines) to get actual closest distances
    - Make sure the loss stays smooth when getting to zero
 - Investigate option for hard(er) collision constraints
    - Look into Lagrange multipliers
 - Implement plotters for .csv statistics