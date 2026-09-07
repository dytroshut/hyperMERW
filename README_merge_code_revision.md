# Corrected merging experiment code

These files implement the power-weighted receiver scaling derived in the
revised manuscript:

- `merge_sinkhorn_uniform_k3.m`: corrected uniform `k=3` solver;
- `merge_sinkhorn_nonuniform_k23.m`: joint `k=2,3` solver with one common
  receiver potential;
- `merge_map_k3.m`: polynomial `k=3` state map;
- `demo_merge_plot.m`: corrected synthetic experiment with the original plot
  layout and weight pairs;
- `test_corrected_merge.m`: regression check against the old saved MAT file.

Copy the five MATLAB files into the same project directory and run
`demo_merge_plot.m` from that directory. It creates:

- `merge_mixing_curves_corrected.mat`;
- `figures/merging_corrected.png` (the script creates the directory if needed).

The old result files are intentionally not overwritten. After inspecting the
corrected figure, either replace the manuscript figure with it or rename it to
`merging.png`.

Run `test_corrected_merge.m` first if you want a short numerical comparison
between the old feasible tensor and the corrected optimizer. That optional test
expects your existing `merge_mixing_curves_safe.mat` in the same directory; the
new demo itself does not require the old MAT file.

The plot styling, weights, initial distribution, horizon, markers, and axes are
preserved. The curve values are regenerated because the corrected tensors are
different. The demo uses the manuscript's direct recursion (`eta=1`), not the
old relaxed update with `eta=0.15`.

The `(0,1)` endpoint is solved as a separate pure `k=3` problem. For all
strictly positive weight pairs, the two layers are inferred jointly with the
same receiver potential. This avoids assigning a fictitious unique `k=2`
optimizer at a zero-weight endpoint.

For every weight pair, the MAT file records the inferred tensors, solver
settings, residuals, objective values, Dobrushin coefficients, contraction
bounds, random seed, and reference tensors. A contraction bound greater than
or equal to one means only that the sufficient theorem does not certify global
convergence; it does not contradict the displayed trajectory.
