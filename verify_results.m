function report = verify_results(root)
%VERIFY_RESULTS Check saved tensors without refitting or changing input data.
if nargin < 1, root = fileparts(mfilename('fullpath')); end
b = load(fullfile(root,'results','broadcast_results_corrected.mat'),'results');
b = b.results;
u = load(fullfile(root,'results','broadcast_uniform_corrected.mat'),'uniform_results');
u = u.uniform_results;
m = load(fullfile(root,'results','merge_mixing_curves_corrected.mat'));
n = numel(b.p);
assert(n == 8 && numel(m.p) == 8);
tolB = 2e-11;
tolM = 2e-9;
assert(abs(sum(b.p)-1) < 1e-14 && abs(sum(m.p)-1) < 1e-14);
assert(all(b.p > 0) && all(m.p > 0));
assert(isequal(b.W,m.W),'The demos must use the same weight sweep.');
assert(nnz(b.A2) == 32 && nnz(b.A3) == 52);
assert(nnz(m.A2) == 28 && nnz(m.A3) == 465, ...
    'Random merging references do not match the reported experiment.');
check_tensor(u.B3,u.A3,[1 3 2],tolB);
assert(max(abs(sum(u.B3,[2 3])-1),[],'all') < tolB);
assert(norm(u.Pproj'*u.p-u.p,1) < tolB);
assert(max(abs(u.Pproj-b.Pproj_all{1}),[],'all') < tolB);
report.broadcastRow = zeros(size(b.W,1),1);
report.broadcastStationarity = report.broadcastRow;
report.broadcastKKT = report.broadcastRow;
report.mergeRow = report.broadcastRow;
report.mergeStationarity = report.broadcastRow;
report.mergeKKT = report.broadcastRow;
report.broadcastFinalError = b.mixCurves(end,:).';
report.mergeFinalError = m.mixCurves(end,:).';
for q = 1:size(b.W,1)
    w2 = b.W(q,1); w3 = b.W(q,2);
    B2 = b.B2_all{q}; B3 = b.B3_all{q}; P = b.Pproj_all{q};
    check_tensor(B3,b.A3,[1 3 2],tolB);
    if w2 > 0, check_tensor(B2,b.A2,[],tolB); else, assert(all(B2 == 0,'all')); end
    assert(max(abs(P-(w2*B2+w3*sum(B3,3))),[],'all') < tolB);
    report.broadcastRow(q) = max(abs(sum(P,2)-1));
    report.broadcastStationarity(q) = norm(P'*b.p-b.p,1);
    assert(report.broadcastRow(q) < tolB && report.broadcastStationarity(q) < tolB);
    % Log-ratio first-order condition, with the constant absorbed in u.
    p = b.p; logu = log(b.u_all{q}); logv = log(b.v_all{q});
    kkt = 0;
    for i = 1:n
        for j = 1:n
            if w2 > 0 && b.A2(i,j) > 0
                kkt = max(kkt,abs(log(B2(i,j)/b.A2(i,j))-log(p(i))-logu(i)-p(i)*logv(j)));
            end
            for ell = 1:n
                if b.A3(i,j,ell) > 0
                    kkt = max(kkt,abs(log(B3(i,j,ell)/b.A3(i,j,ell))-log(p(i))-logu(i) ...
                        -p(i)*(logv(j)+logv(ell))/2));
                end
            end
        end
    end
    report.broadcastKKT(q) = kkt;
    assert(kkt < 1e-10);
    qt = b.p0;
    assert(abs(b.mixCurves(1,q)-norm(qt-b.p,1)) < 1e-14);
    for t = 1:numel(b.time)-1
        qt = P'*qt;
        assert(all(qt >= 0) && abs(sum(qt)-1) < 1e-8);
        assert(abs(b.mixCurves(t+1,q)-norm(qt-b.p,1)) < tolB);
    end
    M2 = m.M2ByWeight{q}; M3 = m.M3ByWeight{q}; fit = m.solverResults{q};
    assert(fit.converged);
    check_tensor(M3,m.A3,[2 1 3],tolM);
    row = max(abs(sum(M3,3)-1),[],'all');
    recv = w3*merging_map(M3,m.p);
    if w2 > 0
        check_tensor(M2,m.A2,[],tolM);
        row = max(row,max(abs(sum(M2,2)-1)));
        recv = recv+w2*M2'*m.p;
    else
        assert(isempty(M2),'An inactive merging layer is not an inferred optimizer.');
    end
    report.mergeRow(q) = row;
    report.mergeStationarity(q) = norm(recv-m.p,1);
    assert(row < tolM && report.mergeStationarity(q) < tolM);
    if w2 > 0
        E2 = log(m.A2)+fit.logR2+fit.omega2*fit.logv.';
        E3 = log(m.A3)+fit.logR3+fit.omega3.*reshape(fit.logv,[1 1 n]);
        kkt = max(abs(log(M2(m.A2>0))-E2(m.A2>0)));
        kkt = max(kkt,max(abs(log(M3(m.A3>0))-E3(m.A3>0))));
        for j = 1:n
            assert(max(abs(fit.K2(:,j).*fit.U2.*exp(fit.omega2*fit.logv(j))-M2(:,j))) < tolM);
            assert(max(abs(fit.K3(:,:,j).*fit.U3.*exp(fit.omega3*fit.logv(j))-M3(:,:,j)),[],'all') < tolM);
        end
    else
        E3 = log(m.A3)+fit.logR+fit.omega.*reshape(fit.logv,[1 1 n]);
        kkt = max(abs(log(M3(m.A3>0))-E3(m.A3>0)));
    end
    report.mergeKKT(q) = kkt;
    assert(kkt < 1e-10);
    delta3 = row_delta(reshape(M3,n*n,n));
    bound = 2*w3*delta3;
    if w2 > 0, bound = bound+w2*row_delta(M2); end
    assert(abs(bound-m.contractionBound(q)) < tolM);
    qt = m.q0;
    assert(abs(m.mixCurves(1,q)-norm(qt-m.p,1)) < 1e-14);
    for t = 1:m.mixSteps
        qtNew = w3*merging_map(M3,qt);
        if w2 > 0, qtNew = qtNew+w2*M2'*qt; end
        assert(all(qtNew >= 0) && abs(sum(qtNew)-1) < tolM);
        qt = qtNew/sum(qtNew);
        assert(abs(m.mixCurves(t+1,q)-norm(qt-m.p,1)) < tolM);
    end
end
report.passed = true;
report.matlabVersion = version;
save(fullfile(root,'results','verification.mat'),'report');
fprintf('\nAll saved-result checks passed.\n');
fprintf('Broadcasting: max row %.3e, stationarity %.3e, KKT %.3e\n', ...
    max(report.broadcastRow),max(report.broadcastStationarity),max(report.broadcastKKT));
fprintf('Merging: max row %.3e, stationarity %.3e, KKT %.3e\n', ...
    max(report.mergeRow),max(report.mergeStationarity),max(report.mergeKKT));
end

function check_tensor(T,A,order,tol)
assert(all(isfinite(T),'all') && all(T >= 0,'all'));
assert(all(T(A == 0) == 0),'The inferred tensor added forbidden entries.');
assert(all(T(A > 0) > 0),'An allowed entry lost positivity.');
if ~isempty(order), assert(max(abs(T-permute(T,order)),[],'all') < tol); end
end

function y = merging_map(M,p)
y = reshape(sum(M.*(p*p.'),[1 2]),[],1);
end

function d = row_delta(T)
d = 0;
for i = 1:size(T,1)
    for j = i+1:size(T,1), d = max(d,sum(abs(T(i,:)-T(j,:)))/2); end
end
end
