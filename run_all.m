function run_all
%RUN_ALL Reproduce the synthetic broadcasting and merging experiments.
% Run from this folder. Existing generated results/figures/README are replaced.
root = fileparts(mfilename('fullpath'));
oldPath = path;
restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(fullfile(root,'broadcasting'),fullfile(root,'merging'));
oldVisibility = get(groot,'defaultFigureVisible');
restoreVisibility = onCleanup(@() set(groot,'defaultFigureVisible',oldVisibility)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off');
fprintf('Running uniform broadcasting...\n');
demo_uniform_b(root);
fprintf('\nRunning non-uniform broadcasting...\n');
demo_nonuniform_b(root);
fprintf('\nRunning merging...\n');
demo_merge_plot(root);
test_solvers;
verify_results(root);
write_readme(root);
fprintf('\nComplete. Results, figures, and README are in %s\n',root);
end
