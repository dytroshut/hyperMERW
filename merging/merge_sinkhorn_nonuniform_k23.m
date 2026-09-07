function out = merge_sinkhorn_nonuniform_k23(A2, A3, p, lambda2, lambda3, varargin)
%MERGE_SINKHORN_NONUNIFORM_K23 Corrected joint merging MERW solve for k=2,3.
%
%   out = merge_sinkhorn_nonuniform_k23(A2,A3,p,lambda2,lambda3,...)
%
% The two layers are inferred jointly.  They share one receiver potential
% v, while their pivot scalings are layer dependent:
%
%   M2(i,j) = A2(i,j) * R2(i) * v(j)^p(i),
%
%   M3(i1,i2,j) = A3(i1,i2,j) * R3(i1,i2) ...
%                  * v(j)^(p(i1)*p(i2)).
%
% The constraints are
%
%   sum_j M2(i,j) = 1,
%   sum_j M3(i1,i2,j) = 1,
%
%   lambda2*M2'*p + lambda3*F_M3(p) = p.
%
% For fixed pivot scalings, every v(j) is obtained from one monotone
% scalar root containing contributions from both active layers.  The
% roots are solved in log(v) for numerical stability.
%
% Both weights must be strictly positive and sum to one.  A zero-weight
% endpoint is a separate single-layer problem and should be solved as such.
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
%   out.M2, out.M3         jointly inferred transition layers
%   out.v, out.logv        shared receiver scaling
%   out.R2, out.R3         absorbed pivot scalings, Rk=omega_k.*Uk
%   out.K2, out.K3         pivot-weighted references, Kk=omega_k.*Ak
%   out.res                [M2 row, M3 row, joint stationarity] residuals
%   out.contractionBound   Lemma sufficient-condition coefficient

    validateattributes(A2, {'numeric'}, {'2d','real','finite','nonnegative'});
    n = size(A2,1);
    if size(A2,2) ~= n
        error('A2 must be n-by-n.');
    end

    validateattributes(A3, {'numeric'}, {'real','finite','nonnegative'});
    if ndims(A3) ~= 3 || size(A3,1) ~= n || ...
            size(A3,2) ~= n || size(A3,3) ~= n
        error('A3 must be n-by-n-by-n and compatible with A2.');
    end

    p = p(:);
    validateattributes(p, {'numeric'}, {'real','finite','positive','numel',n});
    p = p / sum(p);

    validateattributes(lambda2, {'numeric'}, ...
        {'real','finite','scalar','positive'});
    validateattributes(lambda3, {'numeric'}, ...
        {'real','finite','scalar','positive'});
    if abs(lambda2+lambda3-1) > 1e-12
        error('lambda2 and lambda3 must sum to one.');
    end

    opts.maxIter = 5000;
    opts.tol = 1e-10;
    opts.rootTol = 1e-12;
    opts.maxBracketExpansions = 100;
    opts.verbose = true;
    opts.enforceSymmetry = true;
    opts.symmetryTol = 1e-12;
    opts = parse_options(opts,varargin{:});

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

    if any(sum(A2,2) <= 0)
        error('Every k=2 pivot must have at least one supported receiver.');
    end
    rowMass3 = sum(A3,3);
    if any(rowMass3(:) <= 0)
        error('Every k=3 pivot pair must have at least one supported receiver.');
    end

    omega2 = p;
    omega3 = p*p.';

    logA2 = -inf(size(A2));
    support2 = A2 > 0;
    logA2(support2) = log(A2(support2));

    logA3 = -inf(size(A3));
    support3 = A3 > 0;
    logA3(support3) = log(A3(support3));

    logv = zeros(n,1);
    logR2 = zeros(n,1);
    logR3 = zeros(n,n);
    M2 = zeros(n,n);
    M3 = zeros(n,n,n);
    resHist = zeros(opts.maxIter,3);
    converged = false;

    for t = 1:opts.maxIter
        % Exact output-stochastic scaling in each layer.
        for i = 1:n
            terms = reshape(logA2(i,:),[],1) + omega2(i)*logv;
            logR2(i) = -logsumexp_vec(terms);
        end
        for i1 = 1:n
            for i2 = 1:n
                terms = reshape(logA3(i1,i2,:),[],1) ...
                    + omega3(i1,i2)*logv;
                logR3(i1,i2) = -logsumexp_vec(terms);
            end
        end

        % Joint receiver solve: both active layers contribute to Phi_j.
        nextLogv = zeros(n,1);
        for j = 1:n
            logCoeff = zeros(0,1);
            exponent = zeros(0,1);

            active2 = A2(:,j) > 0;
            logCoeff2 = log(lambda2) + log(omega2(active2)) ...
                + logA2(active2,j) + logR2(active2);
            logCoeff = [logCoeff; logCoeff2]; %#ok<AGROW>
            exponent = [exponent; omega2(active2)]; %#ok<AGROW>

            A3j = A3(:,:,j);
            active3 = A3j > 0;
            logA3j = logA3(:,:,j);
            logCoeff3 = log(lambda3) + log(omega3(active3)) ...
                + logA3j(active3) + logR3(active3);
            logCoeff = [logCoeff; logCoeff3]; %#ok<AGROW>
            exponent = [exponent; omega3(active3)]; %#ok<AGROW>

            if isempty(logCoeff)
                error('Receiver %d has no support in any active layer.',j);
            end

            nextLogv(j) = solve_receiver_log_root( ...
                logCoeff, exponent, log(p(j)), logv(j), ...
                opts.rootTol, opts.maxBracketExpansions);
        end
        logv = nextLogv;

        % Gauge normalization preserving both transition tensors.
        gaugeShift = 0.5*(max(logv)+min(logv));
        logv = logv-gaugeShift;
        logR2 = logR2 + omega2*gaugeShift;
        logR3 = logR3 + omega3*gaugeShift;

        for j = 1:n
            M2(:,j) = exp(logA2(:,j) + logR2 + omega2*logv(j));
            M3(:,:,j) = exp(logA3(:,:,j) + logR3 + omega3*logv(j));
        end

        rowErr2 = max(abs(sum(M2,2)-1));
        rowErr3 = max(abs(sum(M3,3)-1),[],'all');
        receiver = lambda2*(M2.'*p) + lambda3*merge_map_k3_local(M3,p);
        stationarityErr = norm(receiver-p,1);
        resHist(t,:) = [rowErr2,rowErr3,stationarityErr];

        if opts.verbose && (t == 1 || mod(t,50) == 0)
            fprintf(['iter %5d, M2 row = %.3e, M3 row = %.3e, ' ...
                'joint stationarity = %.3e\n'], ...
                t,rowErr2,rowErr3,stationarityErr);
        end

        if max(resHist(t,:)) < opts.tol
            resHist = resHist(1:t,:);
            converged = true;
            break;
        end
    end

    if ~converged
        warning('merge_sinkhorn_nonuniform_k23:NoConvergence', ...
            'Maximum iteration count reached; final residual is %.3e.', ...
            max(resHist(end,:)));
    end

    % Match the manuscript convention
    % K^(k)=omega^(k).*A^(k) and R^(k)=omega^(k).*U^(k).
    % This output convention does not change the inferred M2 or M3.
    logU2 = logR2-log(omega2);
    logU3 = logR3-log(omega3);
    R2 = safe_exp_scaling(logR2,'R2');
    R3 = safe_exp_scaling(logR3,'R3');
    v = safe_exp_scaling(logv,'v');
    U2 = safe_exp_scaling(logU2,'U2');
    U3 = safe_exp_scaling(logU3,'U3');
    K2 = A2 .* repmat(omega2,1,n);
    K3 = A3 .* repmat(omega3,1,1,n);

    positive2 = support2 & M2 > 0;
    positive3 = support3 & M3 > 0;
    objective2 = sum(M2(positive2).*log(M2(positive2)./A2(positive2)),'all');
    objective3 = sum(M3(positive3).*log(M3(positive3)./A3(positive3)),'all');

    delta2 = dobrushin_rows(M2);
    delta3 = dobrushin_rows(reshape(M3,n*n,n));
    contractionBound = lambda2*delta2 + 2*lambda3*delta3;
    M3Swap = permute(M3,[2 1 3]);

    out.M2 = M2;
    out.M3 = M3;
    out.R2 = R2;
    out.R3 = R3;
    out.logR2 = logR2;
    out.logR3 = logR3;
    out.U2 = U2;
    out.U3 = U3;
    out.logU2 = logU2;
    out.logU3 = logU3;
    out.v = v;
    out.logv = logv;
    out.K2 = K2;
    out.K3 = K3;
    out.omega2 = omega2;
    out.omega3 = omega3;
    out.lambda2 = lambda2;
    out.lambda3 = lambda3;
    out.p = p;
    out.reference2 = A2;
    out.reference3 = A3;
    out.res = resHist;
    out.iters = size(resHist,1);
    out.finalRes = resHist(end,:);
    out.rowResidual2 = resHist(end,1);
    out.rowResidual3 = resHist(end,2);
    out.stationarityResidual = resHist(end,3);
    out.receiverResidual = max(abs(receiver-p));
    out.phiResidual = out.receiverResidual;
    out.symmetryResidual3 = max(abs(M3-M3Swap),[],'all');
    out.objective2 = objective2;
    out.objective3 = objective3;
    out.objective = lambda2*objective2 + lambda3*objective3;
    out.delta2 = delta2;
    out.delta3 = delta3;
    out.contractionBound = contractionBound;
    out.contractionTolerance = 1e-12;
    out.contractionCertified = converged ...
        && max(resHist(end,:)) < opts.tol ...
        && contractionBound < 1-out.contractionTolerance;
    out.converged = converged;
    out.options = opts;
end


function opts = parse_options(opts,varargin)
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
        warning('merge_sinkhorn_nonuniform_k23:ScalingRange', ...
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


function y = merge_map_k3_local(M3,q)
    n = size(M3,1);
    q = q(:);
    y = zeros(n,1);
    qq = q*q.';
    for j = 1:n
        y(j) = sum(M3(:,:,j).*qq,'all');
    end
end


function coefficient = dobrushin_rows(rows)
    numberRows = size(rows,1);
    coefficient = 0;
    for a = 1:numberRows
        for b = a+1:numberRows
            coefficient = max(coefficient,0.5*sum(abs(rows(a,:)-rows(b,:))));
        end
    end
    coefficient = min(1,max(0,coefficient));
end
