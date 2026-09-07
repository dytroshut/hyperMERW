% Corrected non-uniform k={2,3} broadcasting experiment.
% The k=3 layer is the same topology used in demo_uniform_b. The k=2 layer
% is independently complete within each four-node group and includes all
% self-loops. All weights use the same target p and the same initial p0.

clear; clc; close all;

n = 8;
A3 = build_broadcast_reference(n);
A2 = zeros(n,n);
A2(1:4,1:4) = 1;
A2(5:8,5:8) = 1;
A2 = A2 ./ sum(A2,2);

p = [0.07;0.07;0.07;0.07;0.18;0.18;0.18;0.18];
p = p/sum(p);
p0 = ones(n,1)/n;

W = [
    0.0 1.0
    0.1 0.9
    0.3 0.7
    0.5 0.5
    0.7 0.3
    0.9 0.1
];

mixSteps = 500;
time = (0:mixSteps)';
mixCurves = zeros(mixSteps+1,size(W,1));
Pproj_all = cell(size(W,1),1);
B2_all = cell(size(W,1),1);
B3_all = cell(size(W,1),1);
u_all = cell(size(W,1),1);
v_all = cell(size(W,1),1);
stats = zeros(size(W,1),4); % sweeps, row residual, stationarity residual, gap

for q = 1:size(W,1)
    lambda2 = W(q,1);
    lambda3 = W(q,2);
    fprintf('\nSolving weights (lambda2,lambda3)=(%.1f,%.1f)\n',lambda2,lambda3);
    out = broadcast_sinkhorn_nonuniform_k23(A2,A3,p,lambda2,lambda3, ...
        'maxIter',30000,'tol',1e-13,'scalarTol',1e-15, ...
        'verbose',false,'doMixing',true,'mixSteps',mixSteps,'p0',p0);

    Pproj_all{q} = out.Pproj;
    B2_all{q} = out.B2;
    B3_all{q} = out.B3;
    u_all{q} = out.u;
    v_all{q} = out.v;
    mixCurves(:,q) = out.mix_curve;

    rowErr = norm(out.Pproj*ones(n,1)-ones(n,1),1)/n;
    statErr = norm(out.Pproj'*p-p,1);
    eigAbs = sort(abs(eig(out.Pproj)),'descend');
    spectralGap = 1-eigAbs(2);
    stats(q,:) = [out.iters,rowErr,statErr,spectralGap];
    fprintf('sweeps=%d, row=%.3e, stationarity=%.3e, gap=%.6f\n', ...
        out.iters,rowErr,statErr,spectralGap);
end

initialSpread = max(mixCurves(1,:))-min(mixCurves(1,:));
fprintf('\nMaximum difference among t=0 errors: %.3e\n',initialSpread);

markers = {'o','s','^','d','v','>'};
markerStep = 12;
colors = [
    0.0000 0.4470 0.7410
    0.9290 0.6940 0.1250
    0.4660 0.6740 0.1880
    0.8500 0.1000 0.5500
    0.8500 0.3250 0.0980
    0.4940 0.1840 0.5560
];
figure('Color','w','Position',[100 100 1100 300]); hold on;
for q = 1:size(W,1)
    y = mixCurves(:,q);
    h = plot(time,y,'LineWidth',1.4,'Color',colors(q,:));
    markerIndex = 1:markerStep:numel(time);
    plot(time(markerIndex),y(markerIndex),markers{q}, ...
        'MarkerSize',5,'LineWidth',1.2,'Color',h.Color, ...
        'MarkerFaceColor','none', ...
        'HandleVisibility','off');
    h.DisplayName = sprintf('(%.1f,%.1f)',W(q,1),W(q,2));
end
xlabel('Time step');
ylabel('$\|\mathbf{p}_t-\mathbf{p}\|_1$','Interpreter','latex');
xlim([0 mixSteps]);
grid off;
ax = gca;
ax.FontSize = 14;
ax.LineWidth = 1;
lgd = legend('Location','eastoutside');
lgd.Box = 'off';
lgd.FontSize = 14;
exportgraphics(gcf,'broadcasting_corrected.png','Resolution',300);

results.W = W;
results.p = p;
results.p0 = p0;
results.A2 = A2;
results.A3 = A3;
results.time = time;
results.Pproj_all = Pproj_all;
results.B2_all = B2_all;
results.B3_all = B3_all;
results.u_all = u_all;
results.v_all = v_all;
results.mixCurves = mixCurves;
results.stats = stats;
save('broadcast_results_corrected.mat','results');

function A3 = build_broadcast_reference(n)
    if n ~= 8
        error('This demonstration is defined for n=8.');
    end
    A3 = zeros(n,n,n);
    groups = {[1 2 3 4],[5 6 7 8]};
    for g = 1:numel(groups)
        triples = nchoosek(groups{g},3);
        for q = 1:size(triples,1)
            triple = triples(q,:);
            for pivotPosition = 1:3
                pivot = triple(pivotPosition);
                receivers = triple([1:pivotPosition-1,pivotPosition+1:3]);
                A3(pivot,receivers(1),receivers(2)) = 1;
                A3(pivot,receivers(2),receivers(1)) = 1;
            end
        end
    end

    A3(4,6,7) = 1;
    A3(4,7,6) = 1;
    A3(6,2,3) = 1;
    A3(6,3,2) = 1;

    for pivot = 1:n
        mass = sum(A3(pivot,:,:),'all');
        if mass <= 0
            error('Every pivot must have nonempty support.');
        end
        A3(pivot,:,:) = A3(pivot,:,:)/mass;
    end
end
