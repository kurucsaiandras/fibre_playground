# Dense Fibre Composite Generation

## TODO list
(not necessarily in importance order)
 - Apply same seed for randomness to be able to compare stuff
 - Try BFGS instead of gradient descent (and compare)
    - Maybe other convergence methods as well
 - Use line segments (or even splines) to get actual closest distances
    - Make sure the loss stays smooth when getting to zero
 - Implement statistical descriptors for evaluation
 - Implement Periodic Boundary Conditions
    - As part of this, sort out the domain size vs radius issue
 - Investigate option for hard(er) collision constraints
    - Look into Lagrange multipliers
 - Implement proper configuration parsing
 - Implement plotters for .csv statistics