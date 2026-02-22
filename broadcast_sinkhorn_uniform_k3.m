function out = broadcast_sinkhorn_uniform_k3(A, p, varargin)
%BROADCAST_SINKHORN_UNIFORM_K3  Broadcasting MERW scaling for k = 3 (uniform).
%
%   out = broadcast_sinkhorn_uniform_k3(A, p, 'maxIter',..., 'tol',..., ...)
%
% Solves (k=3 broadcasting, receiver-symmetric scaling)
%   Pi* = A .* (v \otimes u \otimes u)
% under constraints
%   (1/2) Pi x_{2,3} (1,1) = 1
%   (1/2) Pi x_1 p x_3 1   = p
%
% Inputs
%   A : n-by-n-by-n nonnegative tensor (tail mode 1, receivers modes 2,3)
%   p : n-by-1 target stationary distribution, p > 0, sum(p)=1
%
% Name-value options
%   'maxIter'  : maximum iterations (default 5000)
%   'tol'      : stopping tolerance on max residual (default 1e-10)
%   'verbose'  : print progress (default true)
%   'doMixing' : compute mixing curve under projected kernel (default true)
%   'mixSteps' : number of mixing steps (default 50)
%
% Output struct
%   out.Pi        : scaled tensor Pi*
%   out.v, out.u  : scaling vectors
%   out.res       : residual history [res_pivot, res_stat]
%   out.Pproj     : projected kernel Pproj
%   out.mix_curve : ||p_t - p||_1 under Pproj' recursion (if doMixing)
%   out.iters     : number of iterations performed
%   out.finalRes  : final residuals

    % -----------------------
    % Parse inputs and options
    % -----------------------
    if ndims(A) ~= 3
        error('A must be a 3D tensor n-by-n-by-n.')
    end
    n = size(A,1);
    if size(A,2) ~= n || size(A,3) ~= n
        error('A must be n-by-n-by-n.')
    end
    if any(A(:) < 0)
        error('A must be nonnegative.')
    end

    p = p(:);
    if numel(p) ~= n
        error('p must be length n.')
    end
    if any(p <= 0)
        error('p must be strictly positive (for scaling).')
    end
    p = p / sum(p);

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

    resHist = zeros(opts.maxIter, 2);

    % -----------------------
    % Main Sinkhorn–Schrodinger scaling loop
    % -----------------------
    for t = 1:opts.maxIter

        % ----- mpivot(i1) = (1/2) v(i1) * sum_{i2,i3} A(i1,i2,i3) u(i2) u(i3)
        mpivot = zeros(n,1);
        for i1 = 1:n
            S = 0;
            for i2 = 1:n
                ui2 = u(i2);
                for i3 = 1:n
                    S = S + A(i1,i2,i3) * ui2 * u(i3);
                end
            end
            mpivot(i1) = 0.5 * v(i1) * S;
        end

        % v update enforces mpivot = 1
        v = v .* (1 ./ max(mpivot, realmin));

        % ----- mrecv(i2) = (1/2) u(i2) * sum_{i1,i3} p(i1) A(i1,i2,i3) v(i1) u(i3)
        mrecv = zeros(n,1);
        for i2 = 1:n
            S = 0;
            for i1 = 1:n
                pvi = p(i1) * v(i1);
                for i3 = 1:n
                    S = S + pvi * A(i1,i2,i3) * u(i3);
                end
            end
            mrecv(i2) = 0.5 * u(i2) * S;
        end

        % u update enforces mrecv = p
        u = u .* (p ./ max(mrecv, realmin));

        % ----- residual check (recompute once with updated v,u)
        mpivot2 = zeros(n,1);
        for i1 = 1:n
            S = 0;
            for i2 = 1:n
                ui2 = u(i2);
                for i3 = 1:n
                    S = S + A(i1,i2,i3) * ui2 * u(i3);
                end
            end
            mpivot2(i1) = 0.5 * v(i1) * S;
        end

        mrecv2 = zeros(n,1);
        for i2 = 1:n
            S = 0;
            for i1 = 1:n
                pvi = p(i1) * v(i1);
                for i3 = 1:n
                    S = S + pvi * A(i1,i2,i3) * u(i3);
                end
            end
            mrecv2(i2) = 0.5 * u(i2) * S;
        end

        res_pivot = norm(mpivot2 - ones(n,1), 1) / n;
        res_stat  = norm(mrecv2 - p, 1);
        resHist(t,:) = [res_pivot, res_stat];

        if opts.verbose && (t==1 || mod(t,50)==0)
            fprintf('iter %5d, res_pivot = %.3e, res_stat = %.3e\n', t, res_pivot, res_stat);
        end

        if max(resHist(t,:)) < opts.tol
            resHist = resHist(1:t,:);
            break;
        end
    end

    iters = size(resHist,1);

    % -----------------------
    % Build Pi and projected kernel
    % -----------------------
    Pi = zeros(n,n,n);
    for i1 = 1:n
        vi1 = v(i1);
        for i2 = 1:n
            ui2 = u(i2);
            for i3 = 1:n
                Pi(i1,i2,i3) = A(i1,i2,i3) * vi1 * ui2 * u(i3);
            end
        end
    end

    % Pproj(i1,i2) = (1/2) sum_{i3} Pi(i1,i2,i3)
    Pproj = zeros(n,n);
    for i1 = 1:n
        for i2 = 1:n
            Pproj(i1,i2) = 0.5 * sum(Pi(i1,i2,:));
        end
    end

    % -----------------------
    % Optional mixing curve under projected dynamics
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
    out.Pi        = Pi;
    out.v         = v;
    out.u         = u;
    out.res       = resHist;
    out.Pproj     = Pproj;
    out.mix_curve = mix_curve;
    out.iters     = iters;
    out.finalRes  = resHist(end,:);
end