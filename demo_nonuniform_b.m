clear; clc; close all;

% -----------------------
% Build a non-uniform model with k=2 and k=3 layers
% -----------------------
n = 10;

% k=2 layer: a directed graph adjacency matrix A2
A2 = zeros(n,n);
for i = 1:n
    A2(i, mod(i,n)+1) = 1;         % ring
    if i <= 4
        A2(i, 1:4) = A2(i,1:4) + 0.2;  % small community bias
    end
end

% Normalize A2 so rows sum to 1 (standard Markov reference)
for i = 1:n
    s = sum(A2(i,:));
    if s > 0
        A2(i,:) = A2(i,:) / s;
    end
end

% k=3 layer: broadcasting hyperedges A3(i1,i2,i3)
A3 = zeros(n,n,n);
for i1 = 1:n
    for i2 = 1:n
        for i3 = 1:n
            if i2 == i3
                continue
            end
            % within-group structure for receivers
            if (i1<=4 && i2<=4 && i3<=4)
                A3(i1,i2,i3) = 1;
            elseif (i1>=5 && i2>=5 && i3>=5)
                A3(i1,i2,i3) = 1;
            end
        end
    end
end

% Add a few cross events
A3(2,7,8) = 1;
A3(7,2,3) = 1;

% Normalize A3 so (1/2) sum_{i2,i3} A3(i1,i2,i3) = 1 for each i1
for i1 = 1:n
    s = 0.5 * sum(A3(i1,:,:), 'all');
    if s > 0
        A3(i1,:,:) = A3(i1,:,:) / s;
    end
end

% Target stationary distribution p (non-uniform)
p = ones(n,1)/n;
p(1:4) = 0.06;
p(5:10) = (1 - sum(p(1:4))) / 6;
p = p / sum(p);

% Mixture weights
lambda2 = 0.4;
lambda3 = 0.6;

% -----------------------
% Run non-uniform broadcasting scaling
% -----------------------
out = broadcast_sinkhorn_nonuniform_k23(A2, A3, p, lambda2, lambda3, ...
    'maxIter', 3000, ...
    'tol', 1e-12, ...
    'verbose', true, ...
    'doMixing', true, ...
    'mixSteps', 80);

fprintf('\nNormalized weights: lambda2=%.3f, lambda3=%.3f\n', out.lambda2, out.lambda3);
fprintf('Final residuals [row, stationarity] = [%.3e, %.3e]\n', out.finalRes(1), out.finalRes(2));

% -----------------------
% Plot residuals
% -----------------------
figure;
semilogy(out.res(:,1), 'LineWidth', 1.5); hold on;
semilogy(out.res(:,2), 'LineWidth', 1.5);
grid on; axis tight;
xlabel('Iteration');
ylabel('Residual');
legend('row stochasticity', 'stationarity', 'Location', 'best');
title('Non-uniform broadcasting scaling residuals');

% -----------------------
% Plot projected kernel
% -----------------------
figure;
imagesc(out.Pproj);
axis image;
colorbar;
title('Projected kernel P^{proj} (mixture k=2,3)');

% Check constraints numerically
row_err  = norm(out.Pproj * ones(n,1) - ones(n,1), 1) / n;
stat_err = norm(out.Pproj' * p - p, 1);
fprintf('Check Pproj: row_err = %.3e, stat_err = %.3e\n', row_err, stat_err);

% -----------------------
% Plot mixing curve
% -----------------------
if ~isempty(out.mix_curve)
    figure;
    plot(out.mix_curve, 'LineWidth', 1.5);
    grid on; axis tight;
    xlabel('t');
    ylabel('||p_t - p||_1');
    title('Mixing under projected recursion');
end