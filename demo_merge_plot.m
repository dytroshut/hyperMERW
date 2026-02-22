% ============================================
% demo_merge_mixing_curve_safe.m
% One-shot merging experiment with ONE figure only (mixing curves),
% with numerical safety:
%   - enforce symmetric merging kernel in the solver
%   - simplex projection each step (clip + renormalize)
%   - NaN/Inf detection
%
% Requires:
%   merge_sinkhorn_uniform_k3.m   (fixed version with symmetry enforcement)
%   merge_map_k3.m
% ============================================

clear; clc; close all;

rng(1);
n = 8;

% -----------------------
% 1) Random stationary distribution p (this run only)
% -----------------------
alphaDir = 2;
p = rand(n,1).^(1/alphaDir);
p = p / sum(p);

% -----------------------
% 2) Dense merging hypergraph support A3 (k=3 merging)
% -----------------------
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

% Symmetrize input modes (the model assumes inputs are indistinguishable)
A3 = 0.5 * (A3 + permute(A3,[2,1,3]));

% Ensure every (i1,i2) has at least one output
for i1 = 1:n
    for i2 = 1:n
        if sum(A3(i1,i2,:)) == 0
            A3(i1,i2,randi(n)) = 1;
        end
    end
end

% Normalize each input-pair row over output mode as reference (not required, but stable)
for i1 = 1:n
    for i2 = 1:n
        s = sum(A3(i1,i2,:));
        A3(i1,i2,:) = A3(i1,i2,:) / max(s, realmin);
    end
end

% -----------------------
% 3) Build a different k=2 dynamic M2, then scale to A2'*p = p
% -----------------------
densM2 = 0.60;
M2 = zeros(n,n);
for i = 1:n
    for j = 1:n
        if rand < densM2
            M2(i,j) = 1;
        end
    end
end
for i = 1:n
    M2(i, mod(i,n)+1) = M2(i, mod(i,n)+1) + 2.0;
end
for i = 1:n
    if sum(M2(i,:)) == 0
        M2(i,i) = 1;
    end
end
A2 = make_markov_with_stationary_p(M2, p, 8000, 1e-12);

% -----------------------
% 4) Infer merging kernel Xi for this p (symmetry enforced inside)
% -----------------------
XiOut = merge_sinkhorn_uniform_k3(A3, p, ...
    'maxIter', 6000, ...
    'tol', 1e-12, ...
    'verbose', true, ...
    'enforceSymmetry', true);

Xi = XiOut.Xi;

fprintf('\nScaling residuals [row_err, stat_err] = [%.3e, %.3e]\n', XiOut.finalRes(1), XiOut.finalRes(2));

% Optional: check symmetry and constraints numerically (prints only)
Xi_swap = permute(Xi,[2 1 3]);   % avoid invalid indexing on some MATLAB versions
sym_err = max(abs(Xi(:) - Xi_swap(:)));
fprintf('Symmetry max error ||Xi - Xi^{swap}||_inf = %.3e\n', sym_err);

% -----------------------
% 5) Weight sweep and mixing curves
% Mixed recursion: q_{t+1} = lambda2 * (A2' q_t) + lambda3 * F_Xi(q_t)
% Add relaxation + simplex projection for numerical stability.
% -----------------------
W = [
    0.0 1.0
    0.1 0.9
    0.3 0.7
    0.5 0.5
    0.7 0.3
    0.9 0.1
];

mixSteps = 60;
q0 = ones(n,1)/n;

eta = 0.15;  % smaller is safer

mixCurves = zeros(mixSteps, size(W,1));

for k = 1:size(W,1)
    lambda2 = W(k,1);
    lambda3 = W(k,2);

    qt = project_to_simplex(q0);

    for t = 1:mixSteps
        G  = lambda2 * (A2' * qt) + lambda3 * merge_map_k3(Xi, qt);
        qt = (1-eta) * qt + eta * G;

        % keep on simplex to avoid numerical blow-ups
        qt = project_to_simplex(qt);

        if any(~isfinite(qt))
            fprintf('Non-finite state encountered at weight (%.1f,%.1f), step t=%d\n', lambda2, lambda3, t);
            mixCurves(t:end,k) = NaN;
            break;
        end

        mixCurves(t,k) = norm(qt - p, 1);
    end
end

% -----------------------
% 6) ONE plot only: mixing curves
% -----------------------
markers = {'o','s','^','d','v','>'};
markerStep = 6;

figure;
hold on;
for k = 1:size(W,1)
    y = mixCurves(:,k);
    yplot = max(y, eps);
    h = semilogy(1:mixSteps, yplot, 'LineWidth', 1.2);
    idx = 1:markerStep:mixSteps;
    semilogy(idx, yplot(idx), markers{k}, 'MarkerSize', 5, 'LineWidth', 1.5, ...
        'HandleVisibility','off', 'Color', h.Color);
    h.DisplayName = sprintf('(%.1f,%.1f)', W(k,1), W(k,2));
end

grid off;
axis tight;

xlabel('Time step t');
ylabel('||p_t - p||_1');
%title('Merging: mixing under the mixed recursion');

ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 16;
ax.YLabel.FontSize = 16;
ax.Title.FontSize  = 16;

lgd = legend('Location','eastoutside');
lgd.FontSize = 16;
lgd.Box = 'off';
ax.LineWidth = 1.0;

save('merge_mixing_curves_safe.mat', 'W', 'p', 'A2', 'A3', 'Xi', 'mixCurves', 'mixSteps', 'eta', 'q0');




% Show three output slices Xi(:,:,j). Each should be symmetric in (i1,i2).

Jshow = [1 4 8];   % choose any three outputs you want

figure;
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

for t = 1:3
    j = Jshow(t);
    nexttile;
    imagesc(Xi(:,:,j));
    axis image; colorbar;
    title(sprintf('\\Xi(:,:, %d)', j));
    xlabel('input i_2');
    ylabel('input i_1');
end





% ============================================================
% Helper: project a vector to the simplex (clip negatives and renormalize)
% ============================================================
function q = project_to_simplex(q)
    q = q(:);
    q(~isfinite(q)) = 0;
    q(q < 0) = 0;
    s = sum(q);
    if s <= realmin
        q = ones(size(q)) / numel(q);
    else
        q = q / s;
    end
end

% ============================================================
% Helper: build A2 with prescribed stationary distribution p
% ============================================================
function A2 = make_markov_with_stationary_p(M, p, maxIter, tol)
    n = size(M,1);
    p = p(:); p = p/sum(p);

    B = diag(p) * M;

    a = ones(n,1);
    b = ones(n,1);

    for t = 1:maxIter
        rowS = (diag(a) * B * diag(b)) * ones(n,1);
        a = a .* (p ./ max(rowS, realmin));

        colS = (diag(a) * B * diag(b))' * ones(n,1);
        b = b .* (p ./ max(colS, realmin));

        rowS2 = (diag(a) * B * diag(b)) * ones(n,1);
        colS2 = (diag(a) * B * diag(b))' * ones(n,1);
        res = max(norm(rowS2 - p, 1), norm(colS2 - p, 1));
        if res < tol
            break;
        end
    end

    S = diag(a) * B * diag(b);
    A2 = diag(1./p) * S;

    for i = 1:n
        s = sum(A2(i,:));
        if s > 0
            A2(i,:) = A2(i,:) / s;
        end
    end
end