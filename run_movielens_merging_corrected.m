function metrics = run_movielens_merging_corrected(toyFile, outputFolder)
%RUN_MOVIELENS_MERGING_CORRECTED Train and evaluate the corrected k=3 model.
%
% This is the only file that needs to be called manually.  The trainer and
% evaluator must remain in the same folder as this file.

if nargin < 1 || isempty(toyFile)
    toyFile = ['/Users/anqidong/Desktop/Anqi/Papers/merw/' ...
        'movielens_merging_toy_M500_K5.mat'];
end
if nargin < 2 || isempty(outputFolder)
    outputFolder = fileparts(toyFile);
end

assert(isfile(toyFile), 'MovieLens data file not found: %s', toyFile);
assert(isfolder(outputFolder), 'Output folder not found: %s', outputFolder);

codeFolder = fileparts(mfilename('fullpath'));
addpath(codeFolder);

resultFile = fullfile(outputFolder, ...
    'movielens_merw_merging_result_corrected_k3.mat');
metricsFile = fullfile(outputFolder, ...
    'movielens_merw_merging_metrics_corrected_k3.mat');

train_movielens_merw_merging_corrected_k3(toyFile, resultFile);
metrics = eval_movielens_merw_merging_corrected_k3( ...
    toyFile, resultFile, metricsFile);

end
