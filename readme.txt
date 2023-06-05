Abstract:
We address the problem of making Conformal Prediction (CP) intervals locally adaptive. Most existing methods focus on approximating the object-conditional validity of the intervals by partitioning or re-weighting the calibration set. Our strategy is new and conceptually different. 
Instead of re-weighting the calibration data, we redefine the conformity measure through a trainable change of variables, $A \to \phi_X(A)$, that depends explicitly on the object attributes, $X$. Under certain conditions, e.g. if $\phi_X$ is monotonic in $A$ for any $X$, the transformations produce prediction intervals that are guaranteed to be marginally valid and have $X$-dependent sizes. We describe how to parameterize and train $\phi_X$ to maximize the interval efficiency. Contrary to other CP-aware training methods, the objective function is smooth and can be minimized through standard gradient methods without approximations.

Contents of this directory:
- "COPA23contribution20230605.pdf", most recent version of the document "On Training Locally Adaptive CP".
- experiments, a directory containing the code for reproducing the numerical experiments described in Section 3 of "On Training Locally Adaptive CP".
