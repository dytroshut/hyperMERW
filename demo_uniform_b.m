% ============================================
% demo_broadcast_weights_mix_store.m
% Broadcasting with shared stationary p and shared hypergraph topology A3
% k=2 layer A2 has a different dynamic but the same stationary p
%
% Adds:
% - store all projected transition matrices Pproj for all weights
% - plot Pproj for (0,1) and (0.5,0.5)
% - plot stationary distribution p
% - save everything to a .mat file for later use
%
% Requires:
%   broadcast_sinkhorn_nonuniform_k23.m
% ============================================

clear; clc; close all;

% -----------------------
% 1) Fixed hypergraph topology A3 (k=3 broadcasting)
% -----------------------
n = 8;
A3 = zeros(n,n,n);

% Two communities: {1,2,3,4} and {5,6,7,8}
for i1 = 1:n
    for i2 = 1:n
        for i3 = 1:n
            if i2 == i3
                continue
            end
            if (i1<=4 && i2<=4 && i3<=4)
                A3(i1,i2,i3) = 1;
            elseif (i1>=5 && i2>=5 && i3>=5)
                A3(i1,i2,i3) = 1;
            end
        end
    end
end

% Add a couple cross-community hyperedges
A3(4,6,7) = 1;
A3(6,2,3) = 1;

% Normalize A3 so that for each pivot i1, (1/2) sum_{i2,i3} A3(i1,i2,i3) = 1
for i1 = 1:n
    s = 0.5 * sum(A3(i1,:,:), 'all');
    if s > 0
        A3(i1,:,:) = A3(i1,:,:) / s;
    end
end

% -----------------------
% 2) Fixed stationary distribution p (same for all runs)
% -----------------------
p = ones(n,1)/n;
p(1:4) = 0.07;
p(5:8) = (1 - sum(p(1:4))) / 4;
p = p / sum(p);

% Plot stationary distribution (for the paper you can use bar or stem)
figure;
bar(p, 'FaceAlpha', 0.85);
grid on; axis tight;
xlabel('node index');
ylabel('p(i)');
title('Target stationary distribution p');

% Print p numerically (for copy/paste into appendix if needed)
fprintf('\nStationary distribution p:\n');
disp(p(:)');

% -----------------------
% 3) Build a k=2 support consistent with A3 (optional)
% -----------------------
S2 = zeros(n,n);
for i1 = 1:n
    for i2 = 1:n
        if 0.5 * sum(A3(i1,i2,:)) > 0
            S2(i1,i2) = 1;
        end
    end
end

% -----------------------
% 4) Create a k=2 reference dynamic M2 (different from hypergraph projection)
%    then mask by S2 and scale to enforce A2'*p = p
% -----------------------
M2 = zeros(n,n);

% ring drift
for i = 1:n
    M2(i, mod(i,n)+1) = 1;
end

% community bias
for i = 1:4
    M2(i, 1:4) = M2(i,1:4) + 0.6;
end
for i = 5:8
    M2(i, 5:8) = M2(i,5:8) + 0.6;
end

% one shortcut
M2(2,7) = M2(2,7) + 2.0;

% keep topology consistent if desired
M2 = M2 .* S2;

for i = 1:n
    if sum(M2(i,:)) == 0
        M2(i,i) = 1;
    end
end

A2 = make_markov_with_stationary_p(M2, p, 5000, 1e-12);

fprintf('\nCheck A2:\n');
fprintf('  row_err  = %.3e\n', norm(A2*ones(n,1)-ones(n,1),1)/n);
fprintf('  stat_err = %.3e\n', norm(A2'*p-p,1));

% Optionally show A2 as a heatmap
figure;
imagesc(A2);
axis image; colorbar;
title('k=2 Markov layer A^{(2)} with stationary p');

% -----------------------
% 5) Weight sweep
% -----------------------
W = [
    0.0 1.0
    0.1 0.9
    0.3 0.7
    0.5 0.5
    0.7 0.3
    0.9 0.1
];

mixSteps = 500;
p0 = ones(n,1)/n;

opts.maxIter = 3000;
opts.tol     = 1e-12;

% Storage
Pproj_all   = cell(size(W,1),1);     % store each projected kernel
mixCurves   = zeros(mixSteps, size(W,1));
stats_all   = repmat(struct('row_err',[],'stat_err',[],'spectralGap',[]), size(W,1), 1);
scalings_v  = cell(size(W,1),1);     % store v if you want
scalings_u  = cell(size(W,1),1);     % store u if you want

for k = 1:size(W,1)
    lambda2 = W(k,1);
    lambda3 = W(k,2);

    out = broadcast_sinkhorn_nonuniform_k23(A2, A3, p, lambda2, lambda3, ...
        'maxIter', opts.maxIter, ...
        'tol', opts.tol, ...
        'verbose', false, ...
        'doMixing', false);

    Pproj_all{k}  = out.Pproj;
    scalings_v{k} = out.v;
    scalings_u{k} = out.u;

    % Mixing curve under projected recursion
    pt = p0;
    for t = 1:mixSteps
        pt = out.Pproj' * pt;
        mixCurves(t,k) = norm(pt - p, 1);
    end

    stats_all(k) = projected_kernel_stats(out.Pproj, p);

    fprintf('weights (%.1f,%.1f)  row_err=%.2e  stat_err=%.2e  gap=%.2e\n', ...
        lambda2, lambda3, stats_all(k).row_err, stats_all(k).stat_err, stats_all(k).spectralGap);
end

% -----------------------
% 6) Mixing curves on one plot
% -----------------------
markers = {'o','s','^','d','v','>'};
markerStep = 12;


figure;
hold on;
for k = 1:size(W,1)
    y = mixCurves(:,k);
    h = semilogy(1:mixSteps, y, 'LineWidth', 1.2);
    idx = 1:markerStep:mixSteps;
    semilogy(idx, y(idx), markers{k}, 'MarkerSize', 5, 'LineWidth', 1.5, ...
        'HandleVisibility','off', 'Color', h.Color);
    h.DisplayName = sprintf('(%.1f,%.1f)', W(k,1), W(k,2));
end

% ---- remove grey grid ----
grid off;              % or just delete the "grid on" line

axis tight;

% ---- clearer axis labels (edit wording as you like) ----
xlabel('Time step t');           % instead of 't'
ylabel('||p_t - p||_1');         % you can keep this, or use LaTeX later
%title('Mixing under projected dynamics');  % shorter title usually looks nicer

% ---- change font sizes ----
ax = gca;
ax.FontSize = 14;                % tick label size
ax.XLabel.FontSize = 16;         % x-axis label size
ax.YLabel.FontSize = 16;         % y-axis label size
ax.Title.FontSize  = 16;         % title size

% ---- legend placement and font size ----
lgd = legend('Location','eastoutside');
lgd.FontSize = 16;               % legend text size
lgd.Box = 'off';                 % remove legend border box

% ---- optional: make axes lines a bit thicker for print ----
ax.LineWidth = 1.0;

% ---- optional: if you still want grid but not grey, you can do:
% ax.XGrid = 'off'; ax.YGrid = 'off';        % same effect as grid off
% or use subtle grid:
% grid on; ax.GridAlpha = 0.15;              % lighter grid

% figure;
% hold on;
% for k = 1:size(W,1)
%     y = mixCurves(:,k);
%     h = semilogy(1:mixSteps, y, 'LineWidth', 1.2);
%     idx = 1:markerStep:mixSteps;
%     semilogy(idx, y(idx), markers{k}, 'MarkerSize', 5, 'LineWidth', 1.5, ...
%         'HandleVisibility','off', 'Color', h.Color);
%     h.DisplayName = sprintf('(%.1f,%.1f)', W(k,1), W(k,2));
% end
% grid on; axis tight;
% xlabel('t');
% ylabel('||p_t - p||_1');
% title('Mixing curves under (P^{proj})^\top for different weights');
% legend('Location','eastoutside');

% -----------------------
% 7) Plot Pproj matrices for (0,1) and (0.5,0.5)
% -----------------------
idx_hyper = find(W(:,1)==0 & W(:,2)==1, 1);
idx_half  = find(W(:,1)==0.5 & W(:,2)==0.5, 1);

P_hyper = Pproj_all{idx_hyper};
P_half  = Pproj_all{idx_half};

figure;
subplot(1,2,1);
imagesc(P_hyper);
axis image; colorbar;
title('P^{proj} for (0,1)  hypergraph-only');

subplot(1,2,2);
imagesc(P_half);
axis image; colorbar;
title('P^{proj} for (0.5,0.5)  mixed');

fprintf('\nKey projected-kernel numbers\n');
fprintf('Case (0,1):     row_err=%.3e  stat_err=%.3e  gap=%.3e\n', ...
    stats_all(idx_hyper).row_err, stats_all(idx_hyper).stat_err, stats_all(idx_hyper).spectralGap);
fprintf('Case (0.5,0.5): row_err=%.3e  stat_err=%.3e  gap=%.3e\n', ...
    stats_all(idx_half).row_err, stats_all(idx_half).stat_err, stats_all(idx_half).spectralGap);

% -----------------------
% 8) Save everything for paper plots
% -----------------------
results.W = W;
results.p = p;
results.A2 = A2;
results.A3 = A3;
results.Pproj_all = Pproj_all;
results.mixCurves = mixCurves;
results.stats_all = stats_all;
results.scalings_v = scalings_v;
results.scalings_u = scalings_u;
results.mixSteps = mixSteps;

save('broadcast_results.mat', 'results');
fprintf('\nSaved results to broadcast_results.mat\n');


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
            break
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

% ============================================================
% Helper: scalar diagnostics for Pproj
% ============================================================
function stats = projected_kernel_stats(Pproj, p)
    n = size(Pproj,1);
    p = p(:); p = p/sum(p);

    stats.row_err  = norm(Pproj*ones(n,1) - ones(n,1), 1) / n;
    stats.stat_err = norm(Pproj' * p - p, 1);

    ev = eig(Pproj);
    ev = sort(abs(ev), 'descend');
    if numel(ev) >= 2
        stats.spectralGap = max(0, 1 - ev(2));
    else
        stats.spectralGap = NaN;
    end
end