function out = merge_sinkhorn_uniform_k3(A3, p, varargin)
%MERGE_SINKHORN_UNIFORM_K3 Corrected merging MERW scaling for uniform k=3.
%
%   out = merge_sinkhorn_uniform_k3(A3, p, ...)
%
% The corrected KKT factorization is
%
%   M(i1,i2,j) = A3(i1,i2,j) * R(i1,i2) ...
%                * v(j)^(p(i1)*p(i2)).
%
% Here R(i1,i2) absorbs the leading pivot weight in
% K(i1,i2,j)=p(i1)p(i2)A3(i1,i2,j).  It does not remove the
% pivot-dependent exponent on v(j).
%
% The constraints are
%
%   sum_j M(i1,i2,j) = 1,
%   sum_{i1,i2} p(i1)p(i2)M(i1,i2,j) = p(j).
%
% The pivot block is normalized explicitly.  For fixed R, each receiver
% component is obtained from the monotone scalar equation
%
%   Phi_j(s) = sum_{i1,i2} p(i1)p(i2)A3(i1,i2,j)R(i1,i2)
%              s^(p(i1)*p(i2)) = p(j).
%
% Receiver roots are solved in z=log(s), which avoids overflow when some
% pivot weights are small.
%
% Name-value options
%   'maxIter'              default 5000
%   'tol'                  default 1e-10
%   'rootTol'              default 1e-12
%   'maxBracketExpansions' default 100
%   'verbose'              default true
%   'enforceSymmetry'      default true (validate; never alter A3)
%   'symmetryTol'          default 1e-12
%
% Important outputs
%   out.Xi, out.M          corrected transition tensor (optimal if converged)
%   out.v, out.logv        common receiver scaling
%   out.R                  absorbed pivot scaling, R=omega.*U
%   out.U                  theoretical pivot scaling relative to K
%   out.res                [output residual, stationarity residual]
%   out.objective          sum M.*log(M./A3) on the prescribed support

    validateattributes(A3, {'numeric'}, {'real','finite','nonnegative'});
    if ndims(A3) ~= 3
        error('A3 must be a three-dimensional n-by-n-by-n tensor.');
    end

    n = size(A3,1);
    if size(A3,2) ~= n || size(A3,3) ~= n
        error('A3 must be n-by-n-by-n.');
    end

    p = p(:);
    validateattributes(p, {'numeric'}, {'real','finite','positive','numel',n});
    p = p / sum(p);

    opts.maxIter = 5000;
    opts.tol = 1e-10;
    opts.rootTol = 1e-12;
    opts.maxBracketExpansions = 100;
    opts.verbose = true;
    opts.enforceSymmetry = true;
    opts.symmetryTol = 1e-12;
    opts = parse_options(opts, varargin{:});

    validateattributes(opts.maxIter, {'numeric'}, ...
        {'real','finite','scalar','integer','positive'});
    validateattributes(opts.tol, {'numeric'}, {'real','finite','scalar','positive'});
    validateattributes(opts.rootTol, {'numeric'}, {'real','finite','scalar','positive'});
    validateattributes(opts.maxBracketExpansions, {'numeric'}, ...
        {'real','finite','scalar','integer','positive'});
    validateattributes(opts.symmetryTol, {'numeric'}, ...
        {'real','finite','scalar','nonnegative'});
    validate_logical_option(opts.verbose,'verbose');
    validate_logical_option(opts.enforceSymmetry,'enforceSymmetry');
    opts.verbose = logical(opts.verbose);
    opts.enforceSymmetry = logical(opts.enforceSymmetry);

    if opts.enforceSymmetry
        symmetryScale = max(1,max(abs(A3),[],'all'));
        referenceSymmetryError = max( ...
            abs(A3-permute(A3,[2 1 3])),[],'all');
        if referenceSymmetryError > opts.symmetryTol*symmetryScale
            error(['A3 must already be back-symmetric. Symmetrizing inside ' ...
                'the solver would change its support and KL reference.']);
        end
    end

    rowMass = sum(A3,3);
    if any(rowMass(:) <= 0)
        error('Every pivot pair must have at least one supported receiver.');
    end

    omega = p * p.';
    logA3 = -inf(size(A3));
    support = A3 > 0;
    logA3(support) = log(A3(support));

    logv = zeros(n,1);
    logR = zeros(n,n);
    Xi = zeros(n,n,n);
    resHist = zeros(opts.maxIter,2);
    converged = false;

    for t = 1:opts.maxIter
        % Exact pivot-block normalization for the current receiver scaling.
        for i1 = 1:n
            for i2 = 1:n
                terms = reshape(logA3(i1,i2,:),[],1) ...
                    + omega(i1,i2) * logv;
                logR(i1,i2) = -logsumexp_vec(terms);
            end
        end

        % Exact receiver-block solve for fixed R.
        nextLogv = zeros(n,1);
        for j = 1:n
            Aj = A3(:,:,j);
            active = Aj > 0;
            if ~any(active(:))
                error('Receiver %d has no supported incoming pivot pair.',j);
            end

            logAj = logA3(:,:,j);
            logCoeff = log(omega(active)) + logAj(active) + logR(active);
            exponent = omega(active);
            nextLogv(j) = solve_receiver_log_root( ...
                logCoeff, exponent, log(p(j)), logv(j), ...
                opts.rootTol, opts.maxBracketExpansions);
        end
        logv = nextLogv;

        % Fix the one-dimensional scaling gauge without changing Xi.
        gaugeShift = 0.5*(max(logv)+min(logv));
        logv = logv - gaugeShift;
        logR = logR + omega * gaugeShift;

        % Build the transition tensor from the corrected power scaling.
        for j = 1:n
            Xi(:,:,j) = exp(logA3(:,:,j) + logR + omega * logv(j));
        end

        rowErr = max(abs(sum(Xi,3) - 1),[],'all');
        Fp = merge_map_k3_local(Xi,p);
        statErr = norm(Fp-p,1);
        resHist(t,:) = [rowErr, statErr];

        if opts.verbose && (t == 1 || mod(t,50) == 0)
            fprintf(['iter %5d, output residual = %.3e, ' ...
                'stationarity residual = %.3e\n'],t,rowErr,statErr);
        end

        if max(resHist(t,:)) < opts.tol
            resHist = resHist(1:t,:);
            converged = true;
            break;
        end
    end

    if ~converged
        warning('merge_sinkhorn_uniform_k3:NoConvergence', ...
            'Maximum iteration count reached; final residual is %.3e.', ...
            max(resHist(end,:)));
    end

    logU = logR-log(omega);
    R = safe_exp_scaling(logR,'R');
    v = safe_exp_scaling(logv,'v');
    U = safe_exp_scaling(logU,'U');
    K = A3 .* repmat(omega,1,1,n);

    positive = support & Xi > 0;
    objective = sum(Xi(positive) .* log(Xi(positive)./A3(positive)),'all');
    XiSwap = permute(Xi,[2 1 3]);
    symmetryResidual = max(abs(Xi-XiSwap),[],'all');
    delta = dobrushin_rows(reshape(Xi,n*n,n));
    contractionBound = 2*delta;
    contractionTolerance = 1e-12;

    out.Xi = Xi;
    out.M = Xi;
    out.R = R;
    out.logR = logR;
    out.U = U;
    out.logU = logU;
    out.v = v;
    out.logv = logv;
    out.K = K;
    out.omega = omega;
    out.p = p;
    out.reference = A3;
    out.res = resHist;
    out.iters = size(resHist,1);
    out.finalRes = resHist(end,:);
    out.rowResidual = resHist(end,1);
    out.stationarityResidual = resHist(end,2);
    out.receiverResidual = max(abs(Fp-p));
    out.phiResidual = out.receiverResidual;
    out.symmetryResidual = symmetryResidual;
    out.objective = objective;
    out.delta = delta;
    out.contractionBound = contractionBound;
    out.contractionTolerance = contractionTolerance;
    out.contractionCertified = converged ...
        && max(resHist(end,:)) < opts.tol ...
        && contractionBound < 1-contractionTolerance;
    out.converged = converged;
    out.options = opts;
end


function opts = parse_options(opts, varargin)
    if mod(numel(varargin),2) ~= 0
        error('Options must be supplied as name-value pairs.');
    end
    for k = 1:2:numel(varargin)
        name = char(varargin{k});
        if ~isfield(opts,name)
            error('Unknown option "%s".',name);
        end
        opts.(name) = varargin{k+1};
    end
end


function validate_logical_option(value,name)
    if ~isscalar(value) || ...
            ~(islogical(value) || (isnumeric(value) && ismember(value,[0 1])))
        error('Option "%s" must be a scalar logical value.',name);
    end
end


function value = safe_exp_scaling(logValue,name)
    lower = log(realmin('double'));
    upper = log(realmax('double'));
    if any(logValue(:) < lower | logValue(:) > upper)
        warning('merge_sinkhorn_uniform_k3:ScalingRange', ...
            ['Scaling %s exceeds the finite exponentiation range. ' ...
            'Use the returned logarithmic scaling instead.'],name);
    end
    value = exp(logValue);
end


function z = solve_receiver_log_root(logCoeff, exponent, logTarget, z0, rootTol, maxExpand)
    residual = @(x) logsumexp_vec(logCoeff + exponent*x) - logTarget;
    f0 = residual(z0);
    if abs(f0) <= rootTol
        z = z0;
        return;
    end

    step = 1;
    if f0 > 0
        hi = z0;
        lo = z0-step;
        count = 0;
        while residual(lo) > 0 && count < maxExpand
            step = 2*step;
            lo = z0-step;
            count = count+1;
        end
    else
        lo = z0;
        hi = z0+step;
        count = 0;
        while residual(hi) < 0 && count < maxExpand
            step = 2*step;
            hi = z0+step;
            count = count+1;
        end
    end

    if residual(lo) > 0 || residual(hi) < 0
        error('Failed to bracket a receiver root after %d expansions.',maxExpand);
    end

    settings = optimset('TolX',rootTol,'Display','off');
    z = fzero(residual,[lo hi],settings);
end


function value = logsumexp_vec(x)
    x = x(:);
    m = max(x);
    if ~isfinite(m)
        value = m;
    else
        value = m + log(sum(exp(x-m)));
    end
end


function y = merge_map_k3_local(Xi,q)
    n = size(Xi,1);
    q = q(:);
    y = zeros(n,1);
    for j = 1:n
        y(j) = sum(Xi(:,:,j) .* (q*q.'),'all');
    end
end


function coefficient = dobrushin_rows(rows)
    numberRows = size(rows,1);
    coefficient = 0;
    for a = 1:numberRows
        for b = a+1:numberRows
            coefficient = max(coefficient, ...
                0.5*sum(abs(rows(a,:)-rows(b,:))));
        end
    end
    coefficient = min(1,max(0,coefficient));
end
