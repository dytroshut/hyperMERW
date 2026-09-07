function out = broadcast_sinkhorn_uniform_k3(A, p, varargin)
%BROADCAST_SINKHORN_UNIFORM_K3 Corrected broadcasting KL scaling for k=3.
%
% The front-symmetric broadcasting tensor is parametrized as
%   B(i,j,l) = K(i,j,l) * u(i) ...
%              * v(j)^(p(i)/2) * v(l)^(p(i)/2),
%   K(i,j,l) = p(i) * A(i,j,l).
%
% It satisfies
%   sum_{j,l} B(i,j,l)       = 1,
%   sum_{i,l} p(i) B(i,j,l) = p(j).
%
% The pivot update is an exact multiplicative normalization. Receiver
% coordinates are updated cyclically by monotone scalar solves; the old
% componentwise ratio update is not valid for this parametrization.
%
% Name-value options:
%   maxIter       maximum outer sweeps                 (default 20000)
%   tol           residual tolerance                   (default 1e-10)
%   scalarTol     scalar receiver-solve tolerance      (default 1e-13)
%   scalarMaxIter bisection iterations per coordinate  (default 120)
%   verbose       print progress                       (default true)

    validateattributes(A, {'numeric'}, {'nonnegative','real','finite'});
    if ndims(A) ~= 3
        error('A must be an n-by-n-by-n tensor.');
    end
    n = size(A,1);
    if size(A,2) ~= n || size(A,3) ~= n
        error('A must be n-by-n-by-n.');
    end
    Aperm = permute(A,[1 3 2]);
    if norm(A(:)-Aperm(:),2) > 1e-12*max(1,norm(A(:),2))
        error('A must be symmetric in the two receiver modes.');
    end

    p = p(:);
    if numel(p) ~= n || any(~isfinite(p)) || any(p <= 0)
        error('p must be a strictly positive finite vector of length n.');
    end
    p = p / sum(p);

    opts.maxIter       = 20000;
    opts.tol           = 1e-10;
    opts.scalarTol     = 1e-13;
    opts.scalarMaxIter = 120;
    opts.verbose       = true;
    opts = parse_options(opts, varargin{:});

    K = A .* reshape(p,[n 1 1]);
    u = ones(n,1);
    v = ones(n,1);
    resHist = zeros(opts.maxIter,2);
    converged = false;

    for sweep = 1:opts.maxIter
        B = build_tensor(K,p,u,v);
        rowMarg = squeeze(sum(sum(B,3),2));
        if any(rowMarg <= 0)
            error('The support leaves at least one pivot row empty.');
        end
        u = u ./ rowMarg;

        % Cyclic exact receiver projections, using the newest coordinates.
        for j = 1:n
            target = p(j);
            fun = @(s) receiver_coordinate(K,p,u,v,j,s);
            v(j) = solve_monotone(fun,target,v(j),opts.scalarTol,opts.scalarMaxIter);
        end

        % Remove a harmless multiplicative gauge while preserving B.
        [u,v] = normalize_gauge(u,v,p);

        B = build_tensor(K,p,u,v);
        rowMarg = squeeze(sum(sum(B,3),2));
        recvMarg = squeeze(sum(sum(B .* reshape(p,[n 1 1]),3),1));
        recvMarg = recvMarg(:);

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
        warning('broadcast_sinkhorn_uniform_k3:noConvergence', ...
            'Maximum sweep count reached; final residual is %.3e.', ...
            max(resHist(end,:)));
    end

    B = build_tensor(K,p,u,v);
    Pproj = squeeze(sum(B,3));

    out.B = B;
    out.Pi = B; % compatibility with the earlier code
    out.K = K;
    out.u = u;  % pivot scaling
    out.v = v;  % common receiver scaling
    out.Pproj = Pproj;
    out.res = resHist;
    out.iters = size(resHist,1);
    out.converged = converged;
    out.finalRes = resHist(end,:);
end

function B = build_tensor(K,p,u,v)
    n = numel(p);
    B = zeros(n,n,n);
    for i = 1:n
        e = p(i)/2;
        receiverScale = v.^e;
        B(i,:,:) = K(i,:,:) .* u(i) .* ...
            reshape(receiverScale,[1 n 1]) .* reshape(receiverScale,[1 1 n]);
    end
end

function value = receiver_coordinate(K,p,u,v,j,s)
    vTrial = v;
    vTrial(j) = s;
    B = build_tensor(K,p,u,vTrial);
    value = sum(reshape(p,[numel(p) 1]) .* squeeze(B(:,j,:)), 'all');
end

function root = solve_monotone(fun,target,current,scalarTol,maxIter)
    lo = 0;
    hi = max(1,current);
    fhi = fun(hi);
    expansion = 0;
    while fhi < target
        hi = 2*hi;
        expansion = expansion + 1;
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
        mid = lo + (hi-lo)/2;
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
    root = lo + (hi-lo)/2;
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
