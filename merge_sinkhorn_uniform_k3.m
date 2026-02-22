function out = merge_sinkhorn_uniform_k3(A3, p, varargin)
%MERGE_SINKHORN_UNIFORM_K3  Merging MERW scaling for k = 3 (uniform), symmetric inputs.
%
%   out = merge_sinkhorn_uniform_k3(A3, p, 'maxIter',..., 'tol',..., 'verbose',..., 'enforceSymmetry',...)
%
% We compute a merging transition tensor Xi (n-by-n-by-n) of the form
%   Xi(i1,i2,j) = A3(i1,i2,j) * R(i1,i2) * v(j)
% such that
%   (i)  sum_j Xi(i1,i2,j) = 1              for all (i1,i2)  (row-stochastic in output mode)
%   (ii) F_Xi(p) = p,  where F_Xi(q)(j) = sum_{i1,i2} Xi(i1,i2,j) q(i1) q(i2)
%
% If enforceSymmetry is true, we enforce input indistinguishability:
%   Xi(i1,i2,j) = Xi(i2,i1,j).
% This is enforced inside the iteration by symmetrizing A3 and R.
%
% NOTE (MATLAB compatibility):
% Some MATLAB versions do not allow indexing like permute(X,[...])(:).
% This code avoids that pattern.

    % -----------------------
    % Validate inputs
    % -----------------------
    if ndims(A3) ~= 3
        error('A3 must be a 3D tensor n-by-n-by-n.');
    end
    n = size(A3,1);
    if size(A3,2) ~= n || size(A3,3) ~= n
        error('A3 must be n-by-n-by-n.');
    end
    if any(A3(:) < 0)
        error('A3 must be nonnegative.');
    end

    p = p(:);
    if numel(p) ~= n
        error('p must be length n.');
    end
    if any(p <= 0)
        error('p must be strictly positive.');
    end
    p = p / sum(p);

    % -----------------------
    % Options
    % -----------------------
    opts.maxIter = 5000;
    opts.tol     = 1e-10;
    opts.verbose = true;
    opts.enforceSymmetry = true;

    if mod(numel(varargin),2) ~= 0
        error('Options must be name-value pairs.');
    end
    for k = 1:2:numel(varargin)
        name = char(varargin{k});
        val  = varargin{k+1};
        if ~isfield(opts, name)
            error('Unknown option "%s".', name);
        end
        opts.(name) = val;
    end

    % -----------------------
    % Enforce symmetric input support in the reference tensor A3
    % -----------------------
    if opts.enforceSymmetry
        A3 = 0.5 * (A3 + permute(A3,[2 1 3]));
    end

    % -----------------------
    % Initialize scaling
    % -----------------------
    v = ones(n,1);
    R = ones(n,n);
    if opts.enforceSymmetry
        R = 0.5 * (R + R.');
    end

    resHist = zeros(opts.maxIter,2);
    Xi = zeros(n,n,n);

    % -----------------------
    % Main loop
    % -----------------------
    for t = 1:opts.maxIter

        % ---- Step 1: update R to enforce row-stochasticity over output mode
        for i1 = 1:n
            for i2 = 1:n
                denom = 0;
                for j = 1:n
                    denom = denom + A3(i1,i2,j) * v(j);
                end
                R(i1,i2) = 1 / max(denom, realmin);
            end
        end
        if opts.enforceSymmetry
            R = 0.5 * (R + R.');
        end

        % ---- Step 2: update v to enforce stationarity at p
        denomV = zeros(n,1);
        for j = 1:n
            s = 0;
            for i1 = 1:n
                pi1 = p(i1);
                for i2 = 1:n
                    s = s + A3(i1,i2,j) * R(i1,i2) * pi1 * p(i2);
                end
            end
            denomV(j) = s;
        end
        v = p ./ max(denomV, realmin);

        % ---- Build Xi
        for i1 = 1:n
            for i2 = 1:n
                Rij = R(i1,i2);
                for j = 1:n
                    Xi(i1,i2,j) = A3(i1,i2,j) * Rij * v(j);
                end
            end
        end
        if opts.enforceSymmetry
            Xi = 0.5 * (Xi + permute(Xi,[2 1 3]));
        end

        % ---- Residual 1: row-stochasticity
        row_err = 0;
        for i1 = 1:n
            for i2 = 1:n
                s = sum(Xi(i1,i2,:));
                row_err = max(row_err, abs(s - 1));
            end
        end

        % ---- Residual 2: stationarity at p
        Fp = merge_map_k3(Xi, p);
        stat_err = norm(Fp - p, 1);

        resHist(t,:) = [row_err, stat_err];

        if opts.verbose && (t==1 || mod(t,50)==0)
            if opts.enforceSymmetry
                Xi_swap = permute(Xi,[2 1 3]);
                sym_err = max(abs(Xi(:) - Xi_swap(:)));
                fprintf('iter %5d, row_err = %.3e, stat_err = %.3e, sym_err = %.3e\n', t, row_err, stat_err, sym_err);
            else
                fprintf('iter %5d, row_err = %.3e, stat_err = %.3e\n', t, row_err, stat_err);
            end
        end

        if max(resHist(t,:)) < opts.tol
            resHist = resHist(1:t,:);
            break;
        end
    end

    out.Xi       = Xi;
    out.R        = R;
    out.v        = v;
    out.res      = resHist;
    out.iters    = size(resHist,1);
    out.finalRes = resHist(end,:);
end