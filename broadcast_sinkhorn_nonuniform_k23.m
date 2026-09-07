function out = broadcast_sinkhorn_nonuniform_k23(A2,A3,p,lambda2,lambda3,varargin)
%BROADCAST_SINKHORN_NONUNIFORM_K23 Corrected joint scaling for k=2 and k=3.
%
% With K2(i,j)=p(i)A2(i,j) and K3(i,j,l)=p(i)A3(i,j,l),
%   B2(i,j)   = K2(i,j)   u(i) v(j)^p(i),
%   B3(i,j,l) = K3(i,j,l) u(i) ...
%                 v(j)^(p(i)/2) v(l)^(p(i)/2).
% The layer weights enter the mixture constraints, not K2 or K3.
%
% The pivot normalization is exact. Each receiver coordinate is then
% solved sequentially from eta_j=p(j); eta is nonlinear and coupled in v.

    validateattributes(A2, {'numeric'}, {'2d','nonnegative','real','finite'});
    n = size(A2,1);
    if size(A2,2) ~= n
        error('A2 must be n-by-n.');
    end
    validateattributes(A3, {'numeric'}, {'nonnegative','real','finite'});
    if ndims(A3) ~= 3 || any(size(A3) ~= [n n n])
        error('A3 must be n-by-n-by-n.');
    end
    A3perm = permute(A3,[1 3 2]);
    if norm(A3(:)-A3perm(:),2) > 1e-12*max(1,norm(A3(:),2))
        error('A3 must be symmetric in its receiver modes.');
    end

    p = p(:);
    if numel(p) ~= n || any(~isfinite(p)) || any(p <= 0)
        error('p must be a strictly positive finite vector of length n.');
    end
    p = p/sum(p);

    if ~isscalar(lambda2) || ~isscalar(lambda3) || ...
            lambda2 < 0 || lambda3 < 0 || lambda2+lambda3 <= 0
        error('Layer weights must be nonnegative and have positive sum.');
    end
    weightSum = lambda2+lambda3;
    lambda2 = lambda2/weightSum;
    lambda3 = lambda3/weightSum;

    opts.maxIter       = 20000;
    opts.tol           = 1e-10;
    opts.scalarTol     = 1e-13;
    opts.scalarMaxIter = 120;
    opts.verbose       = true;
    opts.doMixing      = true;
    opts.mixSteps      = 50;
    opts.p0            = ones(n,1)/n;
    opts = parse_options(opts,varargin{:});

    K2 = A2 .* p;
    K3 = A3 .* reshape(p,[n 1 1]);
    u = ones(n,1);
    v = ones(n,1);
    resHist = zeros(opts.maxIter,2);
    converged = false;

    for sweep = 1:opts.maxIter
        [B2,B3] = build_tensors(K2,K3,p,u,v);
        rowMarg = lambda2*sum(B2,2) + ...
            lambda3*squeeze(sum(sum(B3,3),2));
        if any(rowMarg <= 0)
            error('The active supports leave at least one pivot row empty.');
        end
        u = u ./ rowMarg;

        for j = 1:n
            target = p(j);
            fun = @(s) receiver_coordinate(K2,K3,p,u,v,j,s,lambda2,lambda3);
            v(j) = solve_monotone(fun,target,v(j),opts.scalarTol,opts.scalarMaxIter);
        end

        [u,v] = normalize_gauge(u,v,p);
        [B2,B3] = build_tensors(K2,K3,p,u,v);
        [rowMarg,recvMarg] = mixture_marginals(B2,B3,p,lambda2,lambda3);
        rowErr = norm(rowMarg-1,1)/n;
        statErr = norm(recvMarg-p,1);
        resHist(sweep,:) = [rowErr statErr];

        if opts.verbose && (sweep == 1 || mod(sweep,50) == 0)
            fprintf('sweep %5d, row residual %.3e, stationarity residual %.3e\n', ...
                sweep,rowErr,statErr);
        end
        if max(rowErr,statErr) <= opts.tol
            converged = true;
            resHist = resHist(1:sweep,:);
            break
        end
    end

    if ~converged
        resHist = resHist(1:opts.maxIter,:);
        warning('broadcast_sinkhorn_nonuniform_k23:noConvergence', ...
            'Maximum sweep count reached; final residual is %.3e.', ...
            max(resHist(end,:)));
    end

    [B2,B3] = build_tensors(K2,K3,p,u,v);
    % A zero-weight layer is outside the joint optimization. Return it as
    % identically zero rather than displaying an arbitrary scaled tensor.
    if lambda2 == 0
        B2 = zeros(n,n);
    end
    if lambda3 == 0
        B3 = zeros(n,n,n);
    end
    P2 = B2;
    P3 = squeeze(sum(B3,3));
    Pproj = lambda2*P2 + lambda3*P3;

    mixCurve = [];
    if opts.doMixing
        p0 = opts.p0(:);
        if numel(p0) ~= n || any(p0 < 0) || sum(p0) <= 0
            error('p0 must be a nonnegative vector of length n and positive mass.');
        end
        pt = p0/sum(p0);
        mixCurve = zeros(opts.mixSteps+1,1);
        mixCurve(1) = norm(pt-p,1);
        for t = 1:opts.mixSteps
            pt = Pproj'*pt;
            mixCurve(t+1) = norm(pt-p,1);
        end
    end

    out.B2 = B2;
    out.B3 = B3;
    out.Pi2 = B2; % compatibility aliases
    out.Pi3 = B3;
    out.K2 = K2;
    out.K3 = K3;
    out.u = u;  % pivot scaling
    out.v = v;  % common receiver scaling
    out.P2 = P2;
    out.P3 = P3;
    out.Pproj = Pproj;
    out.res = resHist;
    out.mix_curve = mixCurve;
    out.iters = size(resHist,1);
    out.converged = converged;
    out.finalRes = resHist(end,:);
    out.lambda2 = lambda2;
    out.lambda3 = lambda3;
end

function [B2,B3] = build_tensors(K2,K3,p,u,v)
    n = numel(p);
    B2 = zeros(n,n);
    B3 = zeros(n,n,n);
    for i = 1:n
        B2(i,:) = K2(i,:) .* u(i) .* reshape(v.^p(i),[1 n]);
        receiverScale = v.^(p(i)/2);
        B3(i,:,:) = K3(i,:,:) .* u(i) .* ...
            reshape(receiverScale,[1 n 1]) .* reshape(receiverScale,[1 1 n]);
    end
end

function [rowMarg,recvMarg] = mixture_marginals(B2,B3,p,lambda2,lambda3)
    n = numel(p);
    rowMarg = lambda2*sum(B2,2) + ...
        lambda3*squeeze(sum(sum(B3,3),2));
    recv2 = B2'*p;
    recv3 = squeeze(sum(sum(B3 .* reshape(p,[n 1 1]),3),1));
    recvMarg = lambda2*recv2 + lambda3*recv3(:);
end

function value = receiver_coordinate(K2,K3,p,u,v,j,s,lambda2,lambda3)
    vTrial = v;
    vTrial(j) = s;
    [B2,B3] = build_tensors(K2,K3,p,u,vTrial);
    value2 = p'*B2(:,j);
    value3 = sum(reshape(p,[numel(p) 1]) .* squeeze(B3(:,j,:)), 'all');
    value = lambda2*value2 + lambda3*value3;
end

function root = solve_monotone(fun,target,current,scalarTol,maxIter)
    lo = 0;
    hi = max(1,current);
    fhi = fun(hi);
    expansion = 0;
    while fhi < target
        hi = 2*hi;
        expansion = expansion+1;
        if expansion > 1024 || ~isfinite(hi) || hi > realmax/4
            error('Unable to bracket a receiver-coordinate root.');
        end
        fhi = fun(hi);
        if ~isfinite(fhi)
            break
        end
    end
    if fhi < target
        error('Receiver-coordinate marginal does not reach its target.');
    end

    for it = 1:maxIter
        mid = lo+(hi-lo)/2;
        fmid = fun(mid);
        if fmid < target
            lo = mid;
        else
            hi = mid;
        end
        if abs(fmid-target) <= scalarTol*max(1,target) || ...
                (hi-lo) <= scalarTol*max(1,hi)
            break
        end
    end
    root = lo+(hi-lo)/2;
    if root <= 0 || ~isfinite(root)
        error('Receiver-coordinate solve produced a nonpositive scaling.');
    end
end

function [u,v] = normalize_gauge(u,v,p)
    logGauge = mean(log(v));
    if isfinite(logGauge)
        gauge = exp(logGauge);
        v = v/gauge;
        u = u .* gauge.^p;
    end
end

function opts = parse_options(opts,varargin)
    if mod(numel(varargin),2) ~= 0
        error('Options must be name-value pairs.');
    end
    for q = 1:2:numel(varargin)
        name = char(varargin{q});
        if ~isfield(opts,name)
            error('Unknown option "%s".',name);
        end
        opts.(name) = varargin{q+1};
    end
end
