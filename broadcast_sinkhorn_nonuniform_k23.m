function out = broadcast_sinkhorn_nonuniform_k23(A2, A3, p, lambda2, lambda3, varargin)
%BROADCAST_SINKHORN_NONUNIFORM_K23  Broadcasting MERW scaling for k in {2,3}.
%
%   out = broadcast_sinkhorn_nonuniform_k23(A2, A3, p, lambda2, lambda3, ...)
%
% Model
%   Pi2 = A2 .* (v ⊗ u)
%   Pi3 = A3 .* (v ⊗ u ⊗ u)
%
% Projected kernel
%   Pproj = lambda2 * Pi2 + lambda3 * (1/2) * sum_{i3} Pi3(:,:,i3)
%
% Constraints
%   Pproj * 1 = 1
%   Pproj' * p = p
%
% Inputs
%   A2      : n-by-n nonnegative matrix (k=2 layer)
%   A3      : n-by-n-by-n nonnegative tensor (k=3 layer)
%   p       : n-by-1 target stationary distribution (positive, sum=1)
%   lambda2 : nonnegative weight for k=2 layer
%   lambda3 : nonnegative weight for k=3 layer
%
% Name-value options
%   'maxIter'  : maximum iterations (default 5000)
%   'tol'      : stopping tolerance on max residual (default 1e-10)
%   'verbose'  : print progress (default true)
%   'doMixing' : compute mixing curve under projected kernel (default true)
%   'mixSteps' : number of mixing steps (default 50)
%
% Outputs
%   out.Pi2, out.Pi3 : scaled tensors
%   out.v, out.u     : scaling vectors
%   out.Pproj        : projected kernel
%   out.res          : residual history [res_row, res_stat]
%   out.mix_curve    : ||p_t - p||_1 under Pproj' recursion (if doMixing)
%   out.iters        : number of iterations
%   out.finalRes     : final residuals

    % -----------------------
    % Validate inputs
    % -----------------------
    if ndims(A2) ~= 2
        error('A2 must be a matrix n-by-n.')
    end
    n = size(A2,1);
    if size(A2,2) ~= n
        error('A2 must be n-by-n.')
    end
    if any(A2(:) < 0)
        error('A2 must be nonnegative.')
    end

    if ndims(A3) ~= 3
        error('A3 must be a 3D tensor n-by-n-by-n.')
    end
    if size(A3,1) ~= n || size(A3,2) ~= n || size(A3,3) ~= n
        error('A3 must be n-by-n-by-n and compatible with A2.')
    end
    if any(A3(:) < 0)
        error('A3 must be nonnegative.')
    end

    p = p(:);
    if numel(p) ~= n
        error('p must be length n.')
    end
    if any(p <= 0)
        error('p must be strictly positive (for scaling).')
    end
    p = p / sum(p);

    if lambda2 < 0 || lambda3 < 0
        error('lambda2 and lambda3 must be nonnegative.')
    end
    if lambda2 + lambda3 <= 0
        error('lambda2 + lambda3 must be positive.')
    end
    % normalize weights for convenience
    sLam = lambda2 + lambda3;
    lambda2 = lambda2 / sLam;
    lambda3 = lambda3 / sLam;

    % -----------------------
    % Parse options
    % -----------------------
    opts.maxIter  = 5000;
    opts.tol      = 1e-10;
    opts.verbose  = true;
    opts.doMixing = true;
    opts.mixSteps = 50;

    if mod(numel(varargin),2) ~= 0
        error('Options must be name-value pairs.')
    end
    for k = 1:2:numel(varargin)
        name = varargin{k};
        val  = varargin{k+1};
        if ~ischar(name) && ~isstring(name)
            error('Option names must be strings.')
        end
        name = char(name);
        if ~isfield(opts, name)
            error('Unknown option "%s".', name)
        end
        opts.(name) = val;
    end

    % -----------------------
    % Initialize scaling
    % -----------------------
    v = ones(n,1);
    u = ones(n,1);

    resHist = zeros(opts.maxIter,2);

    % -----------------------
    % Main loop
    % -----------------------
    for t = 1:opts.maxIter

        % ---------- mixture row marginal: mpivot = Pproj * 1 ----------
        mpivot = zeros(n,1);
        for i1 = 1:n
            % k=2 contribution: sum_{i2} A2(i1,i2) * v(i1)*u(i2)
            s2 = 0;
            vi1 = v(i1);
            for i2 = 1:n
                s2 = s2 + A2(i1,i2) * u(i2);
            end
            s2 = vi1 * s2;

            % k=3 contribution: (1/2) sum_{i2,i3} A3(i1,i2,i3) * v(i1)*u(i2)*u(i3)
            s3 = 0;
            for i2 = 1:n
                ui2 = u(i2);
                for i3 = 1:n
                    s3 = s3 + A3(i1,i2,i3) * ui2 * u(i3);
                end
            end
            s3 = 0.5 * vi1 * s3;

            mpivot(i1) = lambda2 * s2 + lambda3 * s3;
        end

        % v update enforces mpivot = 1
        v = v .* (1 ./ max(mpivot, realmin));

        % ---------- mixture stationarity marginal: mrecv = Pproj' * p ----------
        mrecv = zeros(n,1);
        for i2 = 1:n
            % k=2 contribution: sum_{i1} p(i1)*A2(i1,i2)*v(i1)*u(i2)
            s2 = 0;
            for i1 = 1:n
                s2 = s2 + p(i1) * A2(i1,i2) * v(i1);
            end
            s2 = u(i2) * s2;

            % k=3 contribution: (1/2) sum_{i1,i3} p(i1)*A3(i1,i2,i3)*v(i1)*u(i2)*u(i3)
            s3 = 0;
            for i1 = 1:n
                pvi = p(i1) * v(i1);
                for i3 = 1:n
                    s3 = s3 + pvi * A3(i1,i2,i3) * u(i3);
                end
            end
            s3 = 0.5 * u(i2) * s3;

            mrecv(i2) = lambda2 * s2 + lambda3 * s3;
        end

        % u update enforces mrecv = p
        u = u .* (p ./ max(mrecv, realmin));

        % ---------- residual check ----------
        mpivot2 = zeros(n,1);
        mrecv2  = zeros(n,1);

        for i1 = 1:n
            s2 = 0;
            vi1 = v(i1);
            for i2 = 1:n
                s2 = s2 + A2(i1,i2) * u(i2);
            end
            s2 = vi1 * s2;

            s3 = 0;
            for i2 = 1:n
                ui2 = u(i2);
                for i3 = 1:n
                    s3 = s3 + A3(i1,i2,i3) * ui2 * u(i3);
                end
            end
            s3 = 0.5 * vi1 * s3;

            mpivot2(i1) = lambda2 * s2 + lambda3 * s3;
        end

        for i2 = 1:n
            s2 = 0;
            for i1 = 1:n
                s2 = s2 + p(i1) * A2(i1,i2) * v(i1);
            end
            s2 = u(i2) * s2;

            s3 = 0;
            for i1 = 1:n
                pvi = p(i1) * v(i1);
                for i3 = 1:n
                    s3 = s3 + pvi * A3(i1,i2,i3) * u(i3);
                end
            end
            s3 = 0.5 * u(i2) * s3;

            mrecv2(i2) = lambda2 * s2 + lambda3 * s3;
        end

        res_row  = norm(mpivot2 - ones(n,1), 1) / n;
        res_stat = norm(mrecv2 - p, 1);
        resHist(t,:) = [res_row, res_stat];

        if opts.verbose && (t==1 || mod(t,50)==0)
            fprintf('iter %5d, res_row = %.3e, res_stat = %.3e\n', t, res_row, res_stat);
        end

        if max(resHist(t,:)) < opts.tol
            resHist = resHist(1:t,:);
            break;
        end
    end

    iters = size(resHist,1);

    % -----------------------
    % Build Pi2, Pi3, and projected kernel
    % -----------------------
    Pi2 = zeros(n,n);
    for i1 = 1:n
        vi1 = v(i1);
        for i2 = 1:n
            Pi2(i1,i2) = A2(i1,i2) * vi1 * u(i2);
        end
    end

    Pi3 = zeros(n,n,n);
    for i1 = 1:n
        vi1 = v(i1);
        for i2 = 1:n
            ui2 = u(i2);
            for i3 = 1:n
                Pi3(i1,i2,i3) = A3(i1,i2,i3) * vi1 * ui2 * u(i3);
            end
        end
    end

    P3proj = zeros(n,n);
    for i1 = 1:n
        for i2 = 1:n
            P3proj(i1,i2) = 0.5 * sum(Pi3(i1,i2,:));
        end
    end

    Pproj = lambda2 * Pi2 + lambda3 * P3proj;

    % -----------------------
    % Optional mixing curve
    % -----------------------
    mix_curve = [];
    if opts.doMixing
        pt = ones(n,1)/n;
        mix_curve = zeros(opts.mixSteps,1);
        for t = 1:opts.mixSteps
            pt = Pproj' * pt;
            mix_curve(t) = norm(pt - p, 1);
        end
    end

    % -----------------------
    % Pack outputs
    % -----------------------
    out.Pi2       = Pi2;
    out.Pi3       = Pi3;
    out.v         = v;
    out.u         = u;
    out.Pproj     = Pproj;
    out.res       = resHist;
    out.mix_curve = mix_curve;
    out.iters     = iters;
    out.finalRes  = resHist(end,:);
    out.lambda2   = lambda2;
    out.lambda3   = lambda3;
end