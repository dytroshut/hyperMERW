function results = demo_merge_plot(outputRoot)
% DEMO_MERGE_PLOT
% Corrected nonuniform k=2,3 merging MERW experiment.
%
% This script preserves the original one-figure presentation, weight pairs,
% markers, axis labels, and initial distribution.  Unlike the old demo, it
% jointly infers M2 and M3 for every strictly positive weight pair using one
% shared receiver potential and the pivot-dependent powers required by the
% corrected KKT factorization.  A zero-weight endpoint is treated as a
% separate single-layer baseline.
%
% Required files on the MATLAB path:
%   merge_sinkhorn_nonuniform_k23.m
%   merge_sinkhorn_uniform_k3.m
%   merge_map_k3.m

codeVersion = 'synthetic-release-1';
runTimestamp = char(datetime('now','TimeZone','UTC'));
matlabVersion = version;
matlabRelease = version('-release');
matlabPlatform = computer;
scriptDirectory = fileparts(mfilename('fullpath'));
addpath(scriptDirectory);
if nargin < 1, outputRoot = fileparts(scriptDirectory); end

rngSeed = 1;
rng(rngSeed,'twister');
initialRngState = rng;

n = 8;

% -------------------------------------------------------------------------
% 1) Reproduce the original nonuniform target distribution.
% -------------------------------------------------------------------------
targetShape = 2;
p = rand(n,1).^(1/targetShape);
p = p/sum(p);

% -------------------------------------------------------------------------
% 2) Reproduce the original dense, pivot-symmetric k=3 reference tensor.
% -------------------------------------------------------------------------
densA3 = 0.75;
A3 = zeros(n,n,n);
for i1 = 1:n
    for i2 = 1:n
        for j = 1:n
            if rand < densA3
                A3(i1,i2,j) = 1;
            end
        end
    end
end

A3 = 0.5*(A3 + permute(A3,[2 1 3]));

% Repair empty rows without breaking pivot symmetry.
for i1 = 1:n
    for i2 = i1:n
        if sum(A3(i1,i2,:)) == 0
            j = randi(n);
            A3(i1,i2,j) = 1;
            A3(i2,i1,j) = 1;
        end
    end
end

% Row-normalize the reference tensor.  This normalization is not required
% by the KL problem, but it keeps the reference scale reproducible.
for i1 = 1:n
    for i2 = 1:n
        A3(i1,i2,:) = A3(i1,i2,:)/sum(A3(i1,i2,:));
    end
end

referenceSymmetryResidual3 = max( ...
    abs(A3-permute(A3,[2 1 3])),[],'all');
referenceRowResidual3 = max(abs(sum(A3,3)-1),[],'all');

% -------------------------------------------------------------------------
% 3) Reproduce the original k=2 reference using an MH construction.
%    In the corrected experiment A2 is a reference layer; the joint solver
%    returns the inferred M2 layer for each weight pair.
% -------------------------------------------------------------------------
densQ = 0.60;
Q = zeros(n,n);
for i = 1:n
    for j = 1:n
        if rand < densQ
            Q(i,j) = 1;
        end
    end
end

for i = 1:n
    Q(i,mod(i,n)+1) = Q(i,mod(i,n)+1)+2;
end
for i = 1:n
    if sum(Q(i,:)) == 0
        Q(i,i) = 1;
    end
    Q(i,:) = Q(i,:)/sum(Q(i,:));
end
A2 = make_markov_with_stationary_p_mh(Q,p);

referenceRowResidual2 = max(abs(sum(A2,2)-1));
referenceStationarityResidual2 = norm(A2.'*p-p,1);
actualDensity2 = nnz(A2)/numel(A2);
actualDensity3 = nnz(A3)/numel(A3);

fprintf('Reference A2 stationarity ||A2''p-p||_1 = %.3e\n', ...
    referenceStationarityResidual2);
fprintf('Actual support densities: A2 = %.4f, A3 = %.4f\n', ...
    actualDensity2,actualDensity3);

% -------------------------------------------------------------------------
% 4) Use the current weight sweep and 50-step plotting horizon.
% -------------------------------------------------------------------------
W = [
    0.0 1.0
    0.1 0.9
    0.3 0.7
    0.5 0.5
    0.7 0.3
    0.9 0.1
];

mixSteps = 50;
q0 = ones(n,1)/n;

solverSettings.maxIter = 6000;
% The outer feasibility tolerance should be looser than the scalar-root
% tolerance; otherwise roundoff can leave an already accurate solution just
% above the stopping threshold.
solverSettings.tol = 1e-10;
solverSettings.rootTol = 1e-13;
solverSettings.maxBracketExpansions = 100;
solverSettings.verbose = false;
solverSettings.enforceSymmetry = true;
solverSettings.symmetryTol = 1e-12;

numberWeights = size(W,1);
timeSteps = (0:mixSteps).';
initialError = norm(q0-p,1);
mixCurves = zeros(mixSteps+1,numberWeights);
mixCurves(1,:) = initialError;
M2ByWeight = cell(numberWeights,1);
M3ByWeight = cell(numberWeights,1);
solverResults = cell(numberWeights,1);
solverIterations = zeros(numberWeights,1);
rowResidual2 = nan(numberWeights,1);
rowResidual3 = nan(numberWeights,1);
stationarityResidual = nan(numberWeights,1);
receiverResidual = nan(numberWeights,1);
phiResidual = nan(numberWeights,1); % backward-compatible alias
delta2 = nan(numberWeights,1);
delta3 = nan(numberWeights,1);
contractionBound = nan(numberWeights,1);
contractionCertified = false(numberWeights,1);
objective = nan(numberWeights,1);
solverSeconds = nan(numberWeights,1);

totalTimer = tic;

for k = 1:numberWeights
    lambda2 = W(k,1);
    lambda3 = W(k,2);

    fprintf('\nSolving weights (lambda2,lambda3)=(%.1f,%.1f)\n', ...
        lambda2,lambda3);

    solverTimer = tic;
    if lambda2 == 0
        % This is a pure k=3 baseline. There is no uniquely inferred M2.
        fit = merge_sinkhorn_uniform_k3(A3,p, ...
            'maxIter',solverSettings.maxIter, ...
            'tol',solverSettings.tol, ...
            'rootTol',solverSettings.rootTol, ...
            'maxBracketExpansions',solverSettings.maxBracketExpansions, ...
            'verbose',solverSettings.verbose, ...
            'enforceSymmetry',solverSettings.enforceSymmetry, ...
            'symmetryTol',solverSettings.symmetryTol);

        if ~fit.converged
            error('Uniform k=3 solver failed to converge for weight row %d.',k);
        end

        M2 = [];
        M3 = fit.M;
        rowResidual3(k) = fit.rowResidual;
        stationarityResidual(k) = fit.stationarityResidual;
        receiverResidual(k) = fit.receiverResidual;
        phiResidual(k) = fit.receiverResidual;
        delta3(k) = fit.delta;
        contractionBound(k) = fit.contractionBound;
        contractionCertified(k) = fit.contractionCertified;
        objective(k) = fit.objective;
    elseif lambda3 == 0
        error(['The current reference A2 is a feasible pure k=2 baseline, ' ...
            'but this plot does not include the (1,0) endpoint.']);
    else
        fit = merge_sinkhorn_nonuniform_k23(A2,A3,p,lambda2,lambda3, ...
            'maxIter',solverSettings.maxIter, ...
            'tol',solverSettings.tol, ...
            'rootTol',solverSettings.rootTol, ...
            'maxBracketExpansions',solverSettings.maxBracketExpansions, ...
            'verbose',solverSettings.verbose, ...
            'enforceSymmetry',solverSettings.enforceSymmetry, ...
            'symmetryTol',solverSettings.symmetryTol);

        if ~fit.converged
            error('Joint merging solver failed to converge for weight row %d.',k);
        end

        M2 = fit.M2;
        M3 = fit.M3;
        rowResidual2(k) = fit.rowResidual2;
        rowResidual3(k) = fit.rowResidual3;
        stationarityResidual(k) = fit.stationarityResidual;
        receiverResidual(k) = fit.receiverResidual;
        phiResidual(k) = fit.receiverResidual;
        delta2(k) = fit.delta2;
        delta3(k) = fit.delta3;
        contractionBound(k) = fit.contractionBound;
        contractionCertified(k) = fit.contractionCertified;
        objective(k) = fit.objective;
    end
    solverSeconds(k) = toc(solverTimer);

    M2ByWeight{k} = M2;
    M3ByWeight{k} = M3;
    solverResults{k} = fit;

    solverIterations(k) = fit.iters;

    if isempty(M2)
        fprintf(['iterations=%d, pure-k3 row=%.3e, stationarity=%.3e, ' ...
            'contraction bound=%.6f, time=%.3fs\n'], ...
            solverIterations(k),rowResidual3(k),stationarityResidual(k), ...
            contractionBound(k),solverSeconds(k));
    else
        fprintf(['iterations=%d, row2=%.3e, row3=%.3e, ' ...
            'stationarity=%.3e, contraction bound=%.6f, time=%.3fs\n'], ...
            solverIterations(k),rowResidual2(k),rowResidual3(k), ...
            stationarityResidual(k),contractionBound(k),solverSeconds(k));
    end

    % Direct, undamped recursion from the manuscript:
    % q_{t+1}=lambda2*M2'*q_t+lambda3*F_M3(q_t).
    qt = q0;
    for t = 1:mixSteps
        nextState = zeros(n,1);
        if lambda2 > 0
            nextState = nextState+lambda2*(M2.'*qt);
        end
        if lambda3 > 0
            nextState = nextState+lambda3*merge_map_k3(M3,qt);
        end

        if any(~isfinite(nextState))
            error('Nonfinite state for weight row %d at step %d.',k,t);
        end
        if min(nextState) < -1e-12
            error('Negative probability for weight row %d at step %d.',k,t);
        end

        % Correct only floating-point drift.  No lazy/damped step is used.
        nextState = max(nextState,0);
        stateMass = sum(nextState);
        if stateMass <= 0
            error('Zero state mass for weight row %d at step %d.',k,t);
        end
        qt = nextState/stateMass;
        mixCurves(t+1,k) = norm(qt-p,1);
    end
end
totalSeconds = toc(totalTimer);

% Easy aliases for the pure k=3 result requested by the reviewer.
uniformXi = M3ByWeight{1};
uniformResult = solverResults{1};
uniformContractionBound = contractionBound(1);
uniformContractionCertified = contractionCertified(1);

% -------------------------------------------------------------------------
% 5) Preserve the original one-plot visual design.
% -------------------------------------------------------------------------
markers = {'o','s','^','d','v','>'};
markerStep = 5;

figure('Color','w','Position',[100 100 1100 430]);
hold on;
for k = 1:numberWeights
    y = mixCurves(:,k);
    h = plot(timeSteps,y,'LineWidth',1.2);
    idx = 1:markerStep:numel(timeSteps);
    plot(timeSteps(idx),y(idx),markers{k}, ...
        'MarkerSize',5,'LineWidth',1.5, ...
        'HandleVisibility','off','Color',h.Color);
    h.DisplayName = sprintf('(%.1f,%.1f)',W(k,1),W(k,2));
end

axis tight;
ylim([0 1.05*initialError]);
xlabel('Time step t');
ylabel('$\|\mathbf{p}_t-\mathbf{p}\|_1$','Interpreter','latex');

ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 16;
ax.YLabel.FontSize = 16;
ax.LineWidth = 1.0;

lgd = legend('Location','eastoutside');
lgd.FontSize = 16;
lgd.Box = 'off';

% eta is retained only as metadata compatibility with the old MAT file.
% The corrected experiment uses eta=1, i.e. the manuscript recursion.
eta = 1;

figureDirectory = fullfile(outputRoot,'figures');
if ~isfolder(figureDirectory)
    mkdir(figureDirectory);
end
resultDirectory = fullfile(outputRoot,'results');
if ~isfolder(resultDirectory), mkdir(resultDirectory); end
outputMatFile = fullfile(resultDirectory,'merge_mixing_curves_corrected.mat');
outputFigureFile = fullfile(figureDirectory,'merging_corrected.png');

save(outputMatFile, ...
    'codeVersion','runTimestamp','matlabVersion','matlabRelease', ...
    'matlabPlatform', ...
    'rngSeed','initialRngState','n','targetShape', ...
    'densA3','densQ','Q','W','p','q0','A2','A3', ...
    'actualDensity2','actualDensity3','referenceRowResidual2', ...
    'referenceRowResidual3','referenceStationarityResidual2', ...
    'referenceSymmetryResidual3', ...
    'M2ByWeight','M3ByWeight','solverResults','solverSettings', ...
    'solverIterations','rowResidual2','rowResidual3', ...
    'stationarityResidual','receiverResidual','phiResidual', ...
    'delta2','delta3','contractionBound','contractionCertified', ...
    'objective','solverSeconds','totalSeconds', ...
    'uniformXi','uniformResult','uniformContractionBound', ...
    'uniformContractionCertified','mixCurves','mixSteps','timeSteps', ...
    'initialError','eta');

exportgraphics(gcf,outputFigureFile,'Resolution',300);
results = load(outputMatFile);
end

function A2 = make_markov_with_stationary_p_mh(Q,p)
    n = size(Q,1);
    p = p(:)/sum(p);
    Q(Q < 0) = 0;

    for i = 1:n
        if sum(Q(i,:)) <= realmin
            Q(i,i) = 1;
        end
        Q(i,:) = Q(i,:)/sum(Q(i,:));
    end

    A2 = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if i == j || Q(i,j) <= 0
                continue;
            end
            numerator = p(j)*Q(j,i);
            denominator = p(i)*Q(i,j);
            if denominator <= 0
                acceptance = 1;
            else
                acceptance = min(1,numerator/denominator);
            end
            A2(i,j) = Q(i,j)*acceptance;
        end
    end

    for i = 1:n
        A2(i,i) = max(1-sum(A2(i,:)),0);
        A2(i,:) = A2(i,:)/sum(A2(i,:));
    end
end
