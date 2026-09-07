# Manuscript consistency notes

This package preserves the existing synthetic input data and recomputes the results with the corrected solvers. It does not edit the manuscript.

Two points need to be reflected in the manuscript before claiming exact agreement with this release.

1. **Broadcasting matrix and scaling update.** Use `P_uniform` in the README or `uniform_results.Pproj` in `results/broadcast_uniform_corrected.mat`. The corrected first row begins `[0, 0.327533, 0.327533, 0.344934, ...]` to six decimals. It differs from the current manuscript's row beginning `[0, 0.326, 0.326, 0.347, ...]`. The code includes the pivot-dependent receiver exponents and sequential scalar receiver updates. A proof for the former simultaneous ratio update does not establish convergence of that old algorithm.

2. **Merging topology.** The code reproduces the random merging arrays with 28/64 and 465/512 nonzero entries, the reported merging matrix at weights `(0.5,0.5)`, and the contraction bounds `1.069305` through `1.689073`. Those arrays do not have the broadcasting topology with reversed directions. Keeping these results requires describing the merging reference arrays as supplied, not describing them as the same topology used for broadcasting. Changing to the reversed broadcasting topology would be a different experiment and may require additional feasibility decisions.

The full-precision merging target is also supplied. Its three-decimal manuscript display is rounded and should not be used to reconstruct the numerical input.

All checks in `verify_results.m` pass for the released numerical outputs. This verifies the computed arrays and trajectories, not every theorem or statement in the manuscript.
