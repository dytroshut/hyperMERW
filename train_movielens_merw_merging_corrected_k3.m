function resultFile = train_movielens_merw_merging_corrected_k3(toyFile, resultFile, varargin)
%TRAIN_MOVIELENS_MERW_MERGING_CORRECTED_K3 Train the corrected k=3 model.
%
% This implementation uses one row for each unordered pair i1 <= i2 and
% therefore represents the complete back-symmetric 500-by-500-by-500
% transition tensor without storing both tail orientations.  It implements
% the corrected factorization
%
%   M(r,j) = A(r,j) R(r) exp(p(i1)*p(i2)*logv(j)),
%
% where r denotes the canonical pair (i1,i2).  Off-diagonal rows have
% multiplicity two in the stationary marginal, but the multiplicity is NOT
% included in the receiver exponent.
%
% The data-derived support for a training-seen pair is the original
% Support-A rule: top-B successors of either tail, together with outputs
% observed for that canonical pair.  Tail receivers are added as a sparse
% feasibility backbone.  Unseen pairs use only that backbone.  The latter
% admits the explicit stationary kernel that chooses either tail with
% probability 1/2 (and chooses the single tail on a diagonal row).
%
% Example
%   train_movielens_merw_merging_corrected_k3( ...
%       'movielens_merging_toy_M500_K5.mat', ...
%       'movielens_merw_merging_result_corrected_k3.mat');

if nargin < 1 || isempty(toyFile)
    toyFile = 'movielens_merging_toy_M500_K5.mat';
end
if nargin < 2 || isempty(resultFile)
    resultFile = 'movielens_merw_merging_result_corrected_k3.mat';
end

ip = inputParser;
ip.FunctionName = mfilename;
addParameter(ip, 'NeighborhoodSize', 200, ...
    @(x) isscalar(x) && x >= 1 && x == floor(x));
addParameter(ip, 'SupportFloor', 1e-4, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
addParameter(ip, 'MaxIterations', 200, ...
    @(x) isscalar(x) && x >= 1 && x == floor(x));
addParameter(ip, 'ToleranceL1', 1e-9, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
addParameter(ip, 'ToleranceRelative', 1e-7, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
addParameter(ip, 'ToleranceStep', 1e-8, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
addParameter(ip, 'ReceiverTolerance', 1e-12, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
addParameter(ip, 'MaxReceiverIterations', 50, ...
    @(x) isscalar(x) && x >= 1 && x == floor(x));
addParameter(ip, 'PrintEvery', 5, ...
    @(x) isscalar(x) && x >= 1 && x == floor(x));
parse(ip, varargin{:});
opt = ip.Results;

S = load(toyFile, 'p', 'eventsTrain');
assert(isfield(S, 'p') && isfield(S, 'eventsTrain'), ...
    'The data file must contain p and eventsTrain.');
assert(iscell(S.eventsTrain) && numel(S.eventsTrain) >= 3, ...
    'eventsTrain must contain the k=3 layer.');

p = double(S.p(:));
n = numel(p);
assert(all(isfinite(p)) && all(p > 0), 'p must be strictly positive.');
assert(abs(sum(p) - 1) <= 1e-12, 'p must sum to one.');

Etr = double(S.eventsTrain{3});
assert(isnumeric(Etr) && size(Etr,2) == 3, ...
    'The k=3 training events must be an N-by-3 numeric array.');
assert(all(Etr(:) >= 1 & Etr(:) <= n & Etr(:) == floor(Etr(:))), ...
    'Training event indices must be integers in 1:n.');

tails = sort(Etr(:,1:2), 2);
jtr = Etr(:,3);
numEvents = size(Etr,1);

% ---------------------------------------------------------------------
% Enumerate every unordered pair.  This is the compressed form of the
% complete back-symmetric tensor, not merely a map of observed contexts.
% ---------------------------------------------------------------------
numRows = n * (n + 1) / 2;
pairI = zeros(numRows,1,'uint32');
pairJ = zeros(numRows,1,'uint32');
cursor = 1;
for i = 1:n
    idx = cursor:(cursor + n - i);
    pairI(idx) = uint32(i);
    pairJ(idx) = uint32((i:n).');
    cursor = idx(end) + 1;
end

rowOfEvent = canonical_pair_row(tails(:,1), tails(:,2), n);
trainContextCount = accumarray(rowOfEvent, 1, [numRows,1], @sum, 0);
seenContextMask = trainContextCount > 0;
seenRows = find(seenContextMask);

fprintf('Loaded %s\n', toyFile);
fprintf('n=%d, k=3, train events=%d\n', n, numEvents);
fprintf('Canonical rows: total=%d, observed in training=%d\n', ...
    numRows, numel(seenRows));

% ---------------------------------------------------------------------
% Original Support-A statistics, now pooled over both pair orientations.
% ---------------------------------------------------------------------
C = sparse([Etr(:,1); Etr(:,2)], [jtr; jtr], 1, n, n);

B = min(opt.NeighborhoodSize, n);
neighbors = cell(n,1);
for i = 1:n
    [~, js, values] = find(C(i,:));
    if isempty(js)
        neighbors{i} = zeros(0,1,'uint32');
    else
        [~, order] = sort(values, 'descend');
        order = order(1:min(B, numel(order)));
        neighbors{i} = uint32(js(order).');
    end
end

% Pool observed receivers by canonical context without dynamic per-event
% growth.  Only the 20,677 observed rows receive a nonempty cell.
observedOutput = cell(numRows,1);
[sortedRows, order] = sort(rowOfEvent);
sortedOutput = uint32(jtr(order));
starts = [1; find(diff(sortedRows) ~= 0) + 1];
stops = [starts(2:end) - 1; numEvents];
for g = 1:numel(starts)
    r = sortedRows(starts(g));
    observedOutput{r} = unique(sortedOutput(starts(g):stops(g)));
end

% Store the data-derived support separately from the feasibility backbone.
% This keeps the SeenCtx/SeenEdge evaluation definitions independent of
% the fact that the mathematical tensor contains every canonical row.
dataSupportByRow = cell(numRows,1);
dataNnz = 0;
solverNnz = n * n; % one diagonal tail edge; two for every off-diagonal row
for g = 1:numel(seenRows)
    r = seenRows(g);
    i1 = double(pairI(r));
    i2 = double(pairJ(r));
    Jdata = unique([neighbors{i1}(:); neighbors{i2}(:); ...
        observedOutput{r}(:)]);
    dataSupportByRow{r} = Jdata;
    dataNnz = dataNnz + numel(Jdata);
    baseSize = 1 + double(i1 ~= i2);
    solverNnz = solverNnz + numel(unique([Jdata; pairI(r); pairJ(r)])) ...
        - baseSize;
end

fprintf('Support-A entries on observed rows: %d\n', dataNnz);
fprintf('Solver support entries after the tail backbone: %d\n', solverNnz);

Ai = zeros(solverNnz,1);
Aj = zeros(solverNnz,1);
Av = zeros(solverNnz,1);
isDataEdge = false(solverNnz,1);
cursor = 1;

for r = 1:numRows
    i1 = double(pairI(r));
    i2 = double(pairJ(r));
    Jdata = dataSupportByRow{r};
    if isempty(Jdata)
        Jdata = zeros(0,1,'uint32');
    end
    J = unique([Jdata(:); pairI(r); pairJ(r)]);
    Jd = double(J);
    m = numel(Jd);
    idx = cursor:(cursor + m - 1);

    if seenContextMask(r)
        score = full(C(i1,Jd)) + full(C(i2,Jd));
        score = score(:) + opt.SupportFloor;
    else
        score = ones(m,1);
    end
    score = score / sum(score);

    Ai(idx) = r;
    Aj(idx) = Jd;
    Av(idx) = score;
    isDataEdge(idx) = ismember(J, Jdata);
    cursor = idx(end) + 1;
end

assert(cursor == solverNnz + 1, 'Internal support-size mismatch.');
A3 = sparse(Ai, Aj, Av, numRows, n);
dataSupport3 = sparse(Ai(isDataEdge), Aj(isDataEdge), ...
    true(nnz(isDataEdge),1), numRows, n);
clear Ai Aj Av isDataEdge dataSupportByRow observedOutput neighbors C

rowSumA = full(sum(A3,2));
assert(max(abs(rowSumA - 1)) <= 1e-12, ...
    'The reference kernel is not row-stochastic.');

% Canonical weights.  Multiplicity affects the objective and stationary
% marginal, whereas omega itself is the exponent in the corrected KKT form.
omega = p(double(pairI)) .* p(double(pairJ));
multiplicity = 2 * ones(numRows,1);
multiplicity(pairI == pairJ) = 1;
contextWeight = multiplicity .* omega;
assert(abs(sum(contextWeight) - 1) <= 1e-12, ...
    'Canonical context weights do not sum to one.');

% Explicit feasibility check for the tail-selection backbone.
qBackbone = zeros(n,1);
for r = 1:numRows
    i1 = double(pairI(r));
    i2 = double(pairJ(r));
    if i1 == i2
        qBackbone(i1) = qBackbone(i1) + contextWeight(r);
    else
        qBackbone(i1) = qBackbone(i1) + contextWeight(r) / 2;
        qBackbone(i2) = qBackbone(i2) + contextWeight(r) / 2;
    end
end
backboneResidualL1 = norm(qBackbone - p, 1);
assert(backboneResidualL1 <= 1e-12, ...
    'The feasibility-backbone certificate failed.');

% ---------------------------------------------------------------------
% Correct Sinkhorn--Schrodinger block scaling, entirely in log space.
% ---------------------------------------------------------------------
[edgeRow, edgeCol, aValue] = find(A3);
logA = log(aValue);
edgeOmega = omega(edgeRow);
logContextWeight = log(contextWeight);
logp = log(p);

logv = zeros(n,1);
logR = zeros(numRows,1);
history = nan(opt.MaxIterations, 6);
consecutive = 0;
converged = false;

fprintf('Backbone feasibility ||q-p||_1 = %.3e\n', backboneResidualL1);
fprintf('Starting corrected log-domain scaling...\n');

for iteration = 1:opt.MaxIterations
    % Exact row block.
    rowBase = logA + edgeOmega .* logv(edgeCol);
    logR = -grouped_logsumexp(edgeRow, rowBase, numRows);
    logMrow = logA + logR(edgeRow) + edgeOmega .* logv(edgeCol);

    % Exact receiver block.  The scalar roots are independent across j.
    [delta, receiverIterations, receiverResidual] = receiver_log_roots( ...
        edgeRow, edgeCol, edgeOmega, logMrow, logContextWeight, ...
        logp, n, opt.ReceiverTolerance, opt.MaxReceiverIterations);

    logv = logv + delta;
    gaugeShift = p.' * logv;
    logv = logv - gaugeShift;

    % Recompute the exact row normalization for the updated receiver block.
    rowBase = logA + edgeOmega .* logv(edgeCol);
    logR = -grouped_logsumexp(edgeRow, rowBase, numRows);
    logM = logA + logR(edgeRow) + edgeOmega .* logv(edgeCol);
    mValue = exp(logM);

    rowMarginal = accumarray(edgeRow, mValue, [numRows,1], @sum, 0);
    q = accumarray(edgeCol, contextWeight(edgeRow) .* mValue, ...
        [n,1], @sum, 0);

    rowResidual = max(abs(rowMarginal - 1));
    stationaryResidualL1 = norm(q - p, 1);
    stationaryResidualRelative = max(abs(q - p) ./ p);
    centeredDelta = delta - p.' * delta;
    scaledStep = max(abs(edgeOmega .* centeredDelta(edgeCol)));

    history(iteration,:) = [rowResidual, stationaryResidualL1, ...
        stationaryResidualRelative, receiverResidual, scaledStep, ...
        receiverIterations];

    if iteration == 1 || mod(iteration,opt.PrintEvery) == 0
        fprintf(['it=%3d  row=%.3e  stationarity(L1)=%.3e  ' ...
            'stationarity(rel)=%.3e  receiver=%.3e  step=%.3e\n'], ...
            iteration, rowResidual, stationaryResidualL1, ...
            stationaryResidualRelative, receiverResidual, scaledStep);
    end

    passed = rowResidual <= 1e-12 && ...
        stationaryResidualL1 <= opt.ToleranceL1 && ...
        stationaryResidualRelative <= opt.ToleranceRelative && ...
        scaledStep <= opt.ToleranceStep;
    if passed
        consecutive = consecutive + 1;
    else
        consecutive = 0;
    end
    if consecutive >= 2
        converged = true;
        break;
    end
end

history = history(1:iteration,:);
if ~converged
    error(['Corrected k=3 scaling did not converge in %d iterations. ' ...
        'Final stationarity L1 residual: %.3e.'], ...
        opt.MaxIterations, stationaryResidualL1);
end

M3 = sparse(edgeRow, edgeCol, mValue, numRows, n);
objective = sum(multiplicity(edgeRow) .* mValue .* (logM - logA));

diagnostics = struct();
diagnostics.converged = converged;
diagnostics.iterations = iteration;
diagnostics.historyColumns = {'rowInf','stationarityL1', ...
    'stationarityMaxRelative','receiverLogResidual','scaledLogitStep', ...
    'receiverNewtonIterations'};
diagnostics.history = history;
diagnostics.rowResidualInf = rowResidual;
diagnostics.stationarityResidualL1 = stationaryResidualL1;
diagnostics.stationarityResidualMaxRelative = stationaryResidualRelative;
diagnostics.receiverLogResidual = receiverResidual;
diagnostics.scaledLogitStep = scaledStep;
diagnostics.backboneFeasibilityResidualL1 = backboneResidualL1;
diagnostics.objective = objective;
diagnostics.numRows = numRows;
diagnostics.numSeenContexts = nnz(seenContextMask);
diagnostics.numDataSupportEdges = nnz(dataSupport3);
diagnostics.numSolverSupportEdges = nnz(A3);

modelVersion = 'corrected-canonical-full-k3-v1';
supportDescription = [ ...
    'Support-A on canonical training-seen pairs; tail feasibility ' ...
    'backbone on every canonical pair.'];
solverOptions = opt;

save(resultFile, 'modelVersion', 'supportDescription', 'n', 'p', ...
    'pairI', 'pairJ', 'omega', 'multiplicity', 'contextWeight', ...
    'seenContextMask', 'trainContextCount', 'dataSupport3', 'A3', 'M3', ...
    'logR', 'logv', 'solverOptions', 'diagnostics', '-v7.3');

fprintf('Converged in %d iterations.\n', iteration);
fprintf('Final row residual: %.3e\n', rowResidual);
fprintf('Final stationarity ||q-p||_1: %.3e\n', stationaryResidualL1);
fprintf('Saved corrected result: %s\n', resultFile);

end

% =====================================================================
function row = canonical_pair_row(i, j, n)
% Row numbering for (i,j), i<=j, with i outermost and j=i:n.
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

function value = grouped_logsumexp(group, x, numberOfGroups)
maximum = accumarray(group, x, [numberOfGroups,1], @max, -Inf);
scaled = exp(x - maximum(group));
total = accumarray(group, scaled, [numberOfGroups,1], @sum, 0);
assert(all(isfinite(maximum)) && all(total > 0), ...
    'A required row or receiver has empty numerical support.');
value = maximum + log(total);
end

function [delta, iterations, finalResidual] = receiver_log_roots( ...
    edgeRow, edgeCol, edgeOmega, logM, logContextWeight, ...
    logp, n, tolerance, maxIterations)
% Solve, for every receiver j,
%   sum_r b_r M(r,j) exp(omega_r delta_j) = p_j.
% A safeguarded Newton step is kept inside an analytic bracket derived from
% the smallest and largest incoming omega values.

ell = logContextWeight(edgeRow) + logM;
[logq, slope] = grouped_logsumexp_with_slope( ...
    edgeCol, ell, edgeOmega, n);
d = logp - logq;

omegaMin = accumarray(edgeCol, edgeOmega, [n,1], @min, Inf);
omegaMax = accumarray(edgeCol, edgeOmega, [n,1], @max, 0);
assert(all(isfinite(omegaMin)) && all(omegaMin > 0) && all(omegaMax > 0), ...
    'Every receiver must have at least one positive incoming edge.');

boundA = d ./ omegaMin;
boundB = d ./ omegaMax;
lower = min(boundA, boundB);
upper = max(boundA, boundB);
delta = d ./ slope;
delta = min(max(delta, lower), upper);

for iterations = 1:maxIterations
    x = ell + edgeOmega .* delta(edgeCol);
    [logPhi, slope] = grouped_logsumexp_with_slope( ...
        edgeCol, x, edgeOmega, n);
    residual = logPhi - logp;
    finalResidual = max(abs(residual));
    if finalResidual <= tolerance
        return;
    end

    below = residual < 0;
    above = residual > 0;
    lower(below) = delta(below);
    upper(above) = delta(above);

    candidate = delta - residual ./ slope;
    outside = ~isfinite(candidate) | candidate <= lower | candidate >= upper;
    candidate(outside) = (lower(outside) + upper(outside)) / 2;
    delta = candidate;
end

error(['Receiver root solve failed after %d safeguarded Newton steps; ' ...
    'maximum log residual %.3e.'], maxIterations, finalResidual);
end

function [logTotal, slope] = grouped_logsumexp_with_slope( ...
    group, x, exponent, numberOfGroups)
maximum = accumarray(group, x, [numberOfGroups,1], @max, -Inf);
scaled = exp(x - maximum(group));
total = accumarray(group, scaled, [numberOfGroups,1], @sum, 0);
weighted = accumarray(group, scaled .* exponent, ...
    [numberOfGroups,1], @sum, 0);
assert(all(isfinite(maximum)) && all(total > 0), ...
    'A receiver has empty numerical support.');
logTotal = maximum + log(total);
slope = weighted ./ total;
end
