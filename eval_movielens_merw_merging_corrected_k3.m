function metrics = eval_movielens_merw_merging_corrected_k3( ...
    toyFile, resultFile, metricsFile, varargin)
%EVAL_MOVIELENS_MERW_MERGING_CORRECTED_K3 Evaluate the corrected model.
%
% The experiment keeps the original two subsets and scoring settings:
%   SeenCtx  - the canonical pair occurred in training;
%   SeenEdge - SeenCtx and the realized receiver belongs to the original
%              data-derived Support-A set (before feasibility edges).
%
% The pairwise baseline is a lazy random walk based on the empirical
% last-item/next-item transitions.  MERW candidates are ranked directly by
% the unmodified row of M3.  A receiver outside the data-derived support
% counts as a miss.
%
% Example
%   eval_movielens_merw_merging_corrected_k3( ...
%       'movielens_merging_toy_M500_K5.mat', ...
%       'movielens_merw_merging_result_corrected_k3.mat', ...
%       'movielens_merw_merging_metrics_corrected_k3.mat');

if nargin < 1 || isempty(toyFile)
    toyFile = 'movielens_merging_toy_M500_K5.mat';
end
if nargin < 2 || isempty(resultFile)
    resultFile = 'movielens_merw_merging_result_corrected_k3.mat';
end
if nargin < 3
    metricsFile = 'movielens_merw_merging_metrics_corrected_k3.mat';
end

ip = inputParser;
ip.FunctionName = mfilename;
addParameter(ip, 'Cutoffs', [10 20 30 40 100], ...
    @(x) isnumeric(x) && isvector(x) && all(x >= 1) && ...
    all(x == floor(x)));
parse(ip, varargin{:});
opt = ip.Results;
cutoffs = double(opt.Cutoffs(:).');

S = load(toyFile, 'p', 'eventsTrain', 'eventsTest');
R = load(resultFile, 'modelVersion', 'n', 'p', 'M3', ...
    'seenContextMask', 'dataSupport3', 'diagnostics');

requiredData = {'p','eventsTrain','eventsTest'};
for i = 1:numel(requiredData)
    assert(isfield(S,requiredData{i}), ...
        'The data file is missing %s.', requiredData{i});
end
requiredResult = {'modelVersion','n','p','M3','seenContextMask', ...
    'dataSupport3','diagnostics'};
for i = 1:numel(requiredResult)
    assert(isfield(R,requiredResult{i}), ...
        'The corrected result file is missing %s.', requiredResult{i});
end
assert(strcmp(R.modelVersion,'corrected-canonical-full-k3-v1'), ...
    'This evaluator requires the corrected canonical full-k3 result.');
assert(R.diagnostics.converged, 'The saved training run did not converge.');

p = double(S.p(:));
n = numel(p);
assert(R.n == n && max(abs(double(R.p(:)) - p)) <= 1e-14, ...
    'The data and trained model use different p vectors.');
assert(all(cutoffs <= n), 'Every cutoff must be at most n.');

Etr = double(S.eventsTrain{3});
Ete = double(S.eventsTest{3});
assert(size(Etr,2) == 3 && size(Ete,2) == 3, ...
    'The k=3 events must be numeric N-by-3 arrays.');

% Correct canonical lookup for MERW.  The temporal order is retained only
% for the last-item pairwise baseline below.
testTail = sort(Ete(:,1:2), 2);
rowOfTest = canonical_pair_row(testTail(:,1), testTail(:,2), n);
trueReceiver = Ete(:,3);
numberOfTests = size(Ete,1);

isSeenContext = logical(R.seenContextMask(rowOfTest));
linearEdge = rowOfTest + (trueReceiver - 1) * size(R.dataSupport3,1);
isSeenEdge = isSeenContext & logical(full(R.dataSupport3(linearEdge)));

idxSeenContext = find(isSeenContext);
idxSeenEdge = find(isSeenEdge);

fprintf('Corrected canonical coverage (k=3):\n');
fprintf('  test events:  %d\n', numberOfTests);
fprintf('  SeenCtx:      %d/%d = %.4f\n', numel(idxSeenContext), ...
    numberOfTests, numel(idxSeenContext)/numberOfTests);
fprintf('  SeenEdge:     %d/%d = %.4f of SeenCtx\n', numel(idxSeenEdge), ...
    numel(idxSeenContext), numel(idxSeenEdge)/max(numel(idxSeenContext),1));

% Popularity ranking.
[~, popularityOrder] = sort(p, 'descend');
popularityRank = zeros(n,1);
popularityRank(popularityOrder) = (1:n).';

% Lazy RW based on P(j | temporal last item).  The empirical transition
% counts are row-normalized over the fixed 500-movie candidate set, and a
% self-loop of weight 0.1 is added.
lastTrain = Etr(:,2);
nextTrain = Etr(:,3);
bigramCount = full(sparse(lastTrain, nextTrain, 1, n, n));
rowCount = sum(bigramCount,2);
Pbigram = zeros(n,n);
nonzeroRows = rowCount > 0;
Pbigram(nonzeroRows,:) = bigramCount(nonzeroRows,:) ./ rowCount(nonzeroRows);
Pbigram(~nonzeroRows,:) = repmat(p.', nnz(~nonzeroRows), 1);

lazyStayProbability = 0.1;
Plazy = (1-lazyStayProbability) * Pbigram;
Plazy(1:n+1:end) = Plazy(1:n+1:end) + lazyStayProbability;

[~, lazyOrder] = sort(Plazy, 2, 'descend');
lazyRank = zeros(n,n,'uint16');
for i = 1:n
    lazyRank(i,lazyOrder(i,:)) = uint16(1:n);
end
clear bigramCount Pbigram Plazy lazyOrder

methodNames = {'Popularity','Lazy RW','MERW'};
hitSeenContext = evaluate_subset(idxSeenContext, Ete, rowOfTest, ...
    trueReceiver, p, popularityRank, lazyRank, R.M3, cutoffs, ...
    isSeenEdge);
hitSeenEdge = evaluate_subset(idxSeenEdge, Ete, rowOfTest, ...
    trueReceiver, p, popularityRank, lazyRank, R.M3, cutoffs, ...
    isSeenEdge);

rateSeenContext = hitSeenContext / max(numel(idxSeenContext),1);
rateSeenEdge = hitSeenEdge / max(numel(idxSeenEdge),1);

fprintf('\nSeen-edge test events (N=%d)\n', numel(idxSeenEdge));
print_table_rows(methodNames, rateSeenEdge);
fprintf('\nSeen-context test events (N=%d)\n', numel(idxSeenContext));
print_table_rows(methodNames, rateSeenContext);

metrics = struct();
metrics.modelVersion = R.modelVersion;
metrics.k = 3;
metrics.n = n;
metrics.numberOfTestEvents = numberOfTests;
metrics.numberOfSeenContexts = numel(idxSeenContext);
metrics.numberOfSeenEdges = numel(idxSeenEdge);
metrics.seenContextFraction = numel(idxSeenContext)/numberOfTests;
metrics.seenEdgeGivenContextFraction = ...
    numel(idxSeenEdge)/max(numel(idxSeenContext),1);
metrics.cutoffs = cutoffs;
metrics.methodNames = methodNames;
metrics.hitCountSeenContext = hitSeenContext;
metrics.hitRateSeenContext = rateSeenContext;
metrics.hitCountSeenEdge = hitSeenEdge;
metrics.hitRateSeenEdge = rateSeenEdge;
metrics.contextConvention = 'canonical unordered pair';
metrics.randomWalkContext = 'temporal last item';
metrics.lazyStayProbability = lazyStayProbability;
metrics.predictionRule = 'raw M3 row; unsupported receiver is a miss';
metrics.seenEdgeDefinition = ...
    'realized receiver belongs to unsmoothed data-derived Support-A';
metrics.trainingDiagnostics = R.diagnostics;

if ~isempty(metricsFile)
    save(metricsFile, 'metrics');
    fprintf('\nSaved corrected metrics: %s\n', metricsFile);
end

end

% =====================================================================
function hits = evaluate_subset(indices, Etest, rowOfTest, trueReceiver, ...
    p, popularityRank, lazyRank, M3, cutoffs, isSupportedReceiver)

numberOfMethods = 3;
hits = zeros(numberOfMethods, numel(cutoffs));
n = numel(p);

for s = 1:numel(indices)
    t = indices(s);
    j = trueReceiver(t);

    rankPop = popularityRank(j);
    rankLazy = double(lazyRank(Etest(t,2),j));

    scoreMERW = full(M3(rowOfTest(t),:)).';
    rowTotal = sum(scoreMERW);
    assert(isfinite(rowTotal) && abs(rowTotal - 1) <= 1e-9, ...
        'MERW row %d is not stochastic.', rowOfTest(t));
    scoreMERW = scoreMERW / rowTotal;
    [~, orderMERW] = sort(scoreMERW, 'descend');
    rankMERW = zeros(n,1);
    rankMERW(orderMERW) = (1:n).';

    hits(1,:) = hits(1,:) + (rankPop <= cutoffs);
    hits(2,:) = hits(2,:) + (rankLazy <= cutoffs);
    hits(3,:) = hits(3,:) + ...
        (isSupportedReceiver(t) && rankMERW(j) <= cutoffs);
end
end

function print_table_rows(methodNames, rates)
for m = 1:numel(methodNames)
    fprintf('%-12s', methodNames{m});
    fprintf(' & %.4f', rates(m,:));
    fprintf(' \\\\\n');
end
end

function row = canonical_pair_row(i, j, n)
i = double(i(:));
j = double(j(:));
swap = i > j;
tmp = i(swap);
i(swap) = j(swap);
j(swap) = tmp;
i0 = i - 1;
row = 1 + i0 .* n - i0 .* (i0 - 1) / 2 + (j - i);
row = double(row);
end
