# hyperMERW: synthetic broadcasting and merging experiments

This folder contains the corrected MATLAB code, reproducible inputs, full transition tensors, and mixing curves for the synthetic experiments. The MovieLens experiment is maintained separately in the repository.

The corrected broadcasting matrix is recomputed here. It should replace the earlier displayed matrix when matching the manuscript to this release. See [manuscript consistency notes](MANUSCRIPT_CHECK.md) for the remaining matrix and topology discrepancies.

## Reproduce the results

Open this folder as the MATLAB Current Folder and run:

```matlab
run_all
```

This runs uniform broadcasting, non-uniform broadcasting, and merging, checks the saved results, and regenerates this README with the numerical results and all matrices. It writes only the generated files in this package, including this README. It does not read or overwrite the MovieLens files. The run was tested with MATLAB R2025b on Apple silicon. No additional toolbox or external dataset is required.

To run one experiment:

```matlab
addpath('broadcasting', 'merging')
demo_uniform_b       % Uniform k=3 broadcasting
demo_nonuniform_b    % Joint k=2,3 broadcasting, six weight pairs
demo_merge_plot      % Joint k=2,3 merging, including the pure k=3 endpoint
```

After rerunning individual experiments, use `verify_results` and `write_readme` to update the checks and documentation. Run `test_solvers` for additional small-problem regression tests. Output paths are relative to this package, not to a hard-coded computer directory.

## Files

| File | Purpose |
| --- | --- |
| `run_all.m` | Reproduce all synthetic results and regenerate the README |
| `broadcasting/demo_uniform_b.m` | Construct the broadcasting reference and print the uniform projected kernel |
| `broadcasting/demo_nonuniform_b.m` | Joint broadcasting weight sweep and mixing plot |
| `broadcasting/broadcast_sinkhorn_uniform_k3.m` | Corrected uniform broadcasting solver |
| `broadcasting/broadcast_sinkhorn_nonuniform_k23.m` | Corrected non-uniform broadcasting solver |
| `merging/demo_merge_plot.m` | Generate the seeded merging references, infer both layers, and plot mixing |
| `merging/merge_sinkhorn_uniform_k3.m` | Corrected uniform merging solver |
| `merging/merge_sinkhorn_nonuniform_k23.m` | Corrected joint merging solver |
| `merging/merge_map_k3.m` | Apply the nonlinear merging map |
| `verify_results.m` | Check normalization, stationarity, symmetry, allowed entries, scaling equations, and trajectories |
| `test_solvers.m` | Check additional small problems and invalid-input handling |
| `write_readme.m` | Generate the numerical sections below from the saved MAT files |
| `documentation/README_intro.md` | Editable introduction used by the README generator |
| `results/broadcast_uniform_corrected.mat` | Full uniform broadcasting result in `uniform_results` |
| `results/broadcast_results_corrected.mat` | Full broadcasting weight sweep in `results` |
| `results/merge_mixing_curves_corrected.mat` | Full merging weight sweep and inputs |
| `results/verification.mat` | Numerical verification report |

## Experimental setup

### Broadcasting

The eight within-group triples are

```text
{1,2,3}, {1,2,4}, {1,3,4}, {2,3,4},
{5,6,7}, {5,6,8}, {5,7,8}, {6,7,8}.
```

Each node in a triple broadcasts to the other two. The additional cross-group events are `4 -> {6,7}` and `6 -> {2,3}`. Both receiver orderings are stored, so the third-order reference is symmetric in its last two indices. No pivot is repeated among its receivers. There are 26 directed hyperedge events, represented by 52 nonzero ordered tensor entries.

The pairwise layer contains every directed edge within `{1,2,3,4}` and within `{5,6,7,8}`, including self-loops. It has 32 nonzero entries. References are normalized over the receiver modes. The prescribed distribution is

```matlab
p = [0.07; 0.07; 0.07; 0.07; 0.18; 0.18; 0.18; 0.18];
```

For non-uniform broadcasting, the two layers are inferred jointly for each weight pair. The constraint is on their **weighted combined pivot marginal**, not on each layer's row sum separately. Thus `P2 = B2` and `P3 = sum(B3,3)` are layer contributions. Their combination `P = lambda2*P2 + lambda3*P3` is row-stochastic and satisfies `P'*p = p`. Individual contributions need not be stochastic or preserve `p` separately.

The corrected receiver factors are

```text
K2(i,j)   = p(i)*A2(i,j)
K3(i,j,l) = p(i)*A3(i,j,l)
B2(i,j)   = K2(i,j)*u(i)*v(j)^p(i)
B3(i,j,l) = K3(i,j,l)*u(i)*v(j)^(p(i)/2)*v(l)^(p(i)/2)
```

The weights enter the objective and mixture constraints, not `K2` or `K3`. Pivot normalization is followed by sequential scalar receiver solves using the newest coordinates. The old simultaneous ratio update `v = v.*(p./eta)` is not used. The optimization objective is the layer-weighted sum of `B.*log(B./A)` on allowed entries. A zero-weight layer is excluded from optimization and returned as zero for bookkeeping.

### Merging

The merging experiment uses the seeded random reference arrays that produce the reported densities and transition matrix. **These are not the reversed broadcasting topology.** To reproduce these numbers, a manuscript description should identify the merging arrays supplied here, rather than state that the two experiments use the same hypergraph.

The demo uses `rng(1,'twister')`. It draws eight uniform random numbers, takes their square roots, and normalizes them to obtain `p`. It generates the third-order reference with Bernoulli probability 0.75, averages the two tail orderings, repairs empty receiver rows if needed, and normalizes each row. The pairwise reference comes from a directed proposal generated with probability 0.60 and an added directed cycle, followed by a Metropolis--Hastings correction targeting `p`. The full normalized references are saved, so their entry values as well as their zero patterns are reproducible.

There are 28 nonzero entries out of 64 in `A2` and 465 out of 512 in `A3`. Their ordered-entry densities are 0.4375 and 0.908203125. These are array-entry fractions, not fractions of distinct unordered hyperedges. Averaging the tail orderings can increase the fraction of nonzero entries beyond the initial sampling probability. Repeated tail indices are included in this array-based experiment.

The merging stationary vector shown to three decimal places in the manuscript is rounded. Use the full-precision vector saved with the experiment, rather than rebuilding it from the rounded display.

For positive weights the two transition layers are inferred jointly with a common receiver vector and separate pivot scalings:

```text
M2(i,j)   = A2(i,j)*R2(i)*v(j)^p(i)
M3(i,l,j) = A3(i,l,j)*R3(i,l)*v(j)^(p(i)*p(l))
```

Each merging layer is output-stochastic. Their combined map satisfies

```text
lambda2*M2'*p + lambda3*F_M3(p) = p
F_M3(q)(j) = sum_i sum_l M3(i,l,j)*q(i)*q(l)
```

The absorbed pivot scaling is `R = omega.*U`, with `K = omega.*A`. Receiver equations are solved in logarithmic coordinates. The implementation minimizes the layer-weighted sum of `M.*log(M./A)` on allowed entries. The `(0,1)` endpoint uses the uniform solver and has no inferred pairwise matrix, so `M2ByWeight{1}` is empty.

### Mixing curves and numerical checks

Both experiments use the weights

```matlab
W = [0 1; 0.1 0.9; 0.3 0.7; 0.5 0.5; 0.7 0.3; 0.9 0.1];
```

All curves within an experiment start from `p0 = ones(8,1)/8`, with the initial error recorded at `t=0`. Broadcasting runs for 500 steps with `p_next = P'*p_current`. Merging runs for 50 steps using the direct nonlinear map, with no damping. The merging demo only renormalizes floating-point mass drift after each step. Both figures use a linear error axis, and the merging markers are spaced every five steps.

Broadcasting uses outer tolerance `1e-13` and scalar marginal tolerance `1e-15`. Merging uses outer tolerance `1e-10` and scalar root tolerance `1e-13`. A demo stops with an error if its solver does not meet the requested tolerance. The checks also inspect the maximum row residual, not only the mean broadcasting row residual used by its stopping rule.

The merging sufficient contraction bound is `lambda2*delta2 + 2*lambda3*delta3`, where each `delta` is half the maximum L1 distance between two conditional output rows. A bound greater than or equal to one does not certify global contraction. It does not rule out the observed convergence from the stated initialization. The numerical checks are not a proof of convergence for arbitrary inputs.

![Broadcasting mixing curves](figures/broadcasting_corrected.png)

![Merging mixing curves](figures/merging_corrected.png)

## Load full-precision matrices

```matlab
s = load('results/broadcast_uniform_corrected.mat');
P_uniform = s.uniform_results.Pproj;
B3_uniform = s.uniform_results.B3;

s = load('results/broadcast_results_corrected.mat');
b = s.results;
q = find(all(abs(b.W-[0.5 0.5]) < 1e-12,2));
P_mixed = b.Pproj_all{q};
B2 = b.B2_all{q};
B3 = b.B3_all{q};

m = load('results/merge_mixing_curves_corrected.mat');
q = find(all(abs(m.W-[0.5 0.5]) < 1e-12,2));
M2 = m.M2ByWeight{q};
M3 = m.M3ByWeight{q};
M3_first_slice = M3(:,:,1);
```

All eight receiver slices are included below, not only the first three. The README prints six decimal places for readability. Very small positive entries may therefore display as zero. The MAT files retain double precision and are the authoritative source for computations. Tensor slices use MATLAB indexing `(:,:,j)`. Broadcasting slices have axes `(pivot, first receiver)` with the second receiver fixed. Merging slices have the two tails as axes with the receiver fixed.

The following sections are generated directly from the saved results by `write_readme.m`.
