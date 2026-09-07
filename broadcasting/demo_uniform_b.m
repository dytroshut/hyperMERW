function uniform_results = demo_uniform_b(outputRoot)
% Corrected uniform k=3 broadcasting experiment.
% The support contains the directed events obtained from every three-node
% combination within {1,2,3,4} and {5,6,7,8}: each node in the triple is
% used once as pivot and broadcasts to the other two. Two cross-group
% events, 4->{6,7} and 6->{2,3}, are added. Receiver order is stored
% symmetrically, and no pivot is repeated among its receivers.

scriptDirectory = fileparts(mfilename('fullpath'));
addpath(scriptDirectory);
if nargin < 1, outputRoot = fileparts(scriptDirectory); end
resultDirectory = fullfile(outputRoot,'results');
if ~isfolder(resultDirectory), mkdir(resultDirectory); end

n = 8;
A3 = build_broadcast_reference(n);
p = [0.07;0.07;0.07;0.07;0.18;0.18;0.18;0.18];
p = p/sum(p);

out = broadcast_sinkhorn_uniform_k3(A3,p, ...
    'maxIter',20000,'tol',1e-13,'scalarTol',1e-15,'verbose',true);
assert(out.converged,'Uniform broadcasting solver did not converge.');

fprintf('\nCorrected uniform k=3 projected kernel P:\n');
disp(out.Pproj);
fprintf('ordered support entries = %d\n',nnz(A3));
fprintf('outer sweeps            = %d\n',out.iters);
fprintf('row residual            = %.3e\n', ...
    norm(out.Pproj*ones(n,1)-ones(n,1),1)/n);
fprintf('stationarity residual   = %.3e\n',norm(out.Pproj'*p-p,1));
fprintf('largest diagonal entry  = %.3e\n',max(abs(diag(out.Pproj))));

uniform_results.p = p;
uniform_results.A3 = A3;
uniform_results.B3 = out.B;
uniform_results.Pproj = out.Pproj;
uniform_results.u = out.u;
uniform_results.v = out.v;
uniform_results.res = out.res;
uniform_results.converged = out.converged;
uniform_results.iters = out.iters;
uniform_results.matlabVersion = version;
save(fullfile(resultDirectory,'broadcast_uniform_corrected.mat'),'uniform_results');
end

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
