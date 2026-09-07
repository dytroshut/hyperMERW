function test_solvers
%TEST_SOLVERS Small deterministic solver regressions, without external data.
root = fileparts(mfilename('fullpath'));
oldPath = path;
restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(fullfile(root,'broadcasting'),fullfile(root,'merging'));
n = 3; p = ones(n,1)/n;
A2 = ones(n,n)/n; A3 = ones(n,n,n)/(n*n);
b = broadcast_sinkhorn_uniform_k3(A3,p,'tol',1e-11,'scalarTol',1e-14,'verbose',false);
assert(b.converged && max(abs(b.B-A3),[],'all') < 1e-10);
b23 = broadcast_sinkhorn_nonuniform_k23(A2,A3,p,0.4,0.6, ...
    'tol',1e-11,'scalarTol',1e-14,'verbose',false,'doMixing',false);
assert(b23.converged && max(abs(b23.B2-A2),[],'all') < 1e-10);
assert(max(abs(b23.B3-A3),[],'all') < 1e-10);
% Merging rows normalize over the final mode only.
AM3 = ones(n,n,n)/n;
m = merge_sinkhorn_uniform_k3(AM3,p,'tol',1e-11,'rootTol',1e-14,'verbose',false);
assert(m.converged && max(abs(m.M-AM3),[],'all') < 1e-10);
m23 = merge_sinkhorn_nonuniform_k23(A2,AM3,p,0.4,0.6, ...
    'tol',1e-11,'rootTol',1e-14,'verbose',false);
assert(m23.converged && max(abs(m23.M2-A2),[],'all') < 1e-10);
assert(max(abs(m23.M3-AM3),[],'all') < 1e-10);
% Pure k=2 endpoint of the broadcasting solver: prior already feasible.
p2 = [0.2;0.3;0.5]; AP = repmat(p2.',n,1);
b2 = broadcast_sinkhorn_nonuniform_k23(AP,zeros(n,n,n),p2,1,0, ...
    'tol',1e-11,'scalarTol',1e-14,'verbose',false,'doMixing',false);
assert(b2.converged && max(abs(b2.B2-AP),[],'all') < 1e-10);
assert(all(b2.B3 == 0,'all'));
% Receiver-coupling regression. A simultaneous receiver ratio update
% cycles on this strictly feasible example. Sequential scalar solves
% must instead recover the known feasible entropy optimum.
n = 6; p = [0.2;0.2;0.05;0.05;0.25;0.25];
A = zeros(n,n,n); expected = A;
for i = 1:4
    A(i,5,6) = 0.5; A(i,6,5) = 0.5;
end
for i = 5:6
    A(i,1,2) = 0.25; A(i,2,1) = 0.25;
    A(i,3,4) = 0.25; A(i,4,3) = 0.25;
end
expected = A;
for i = 5:6
    expected(i,1,2) = 0.4; expected(i,2,1) = 0.4;
    expected(i,3,4) = 0.1; expected(i,4,3) = 0.1;
end
b = broadcast_sinkhorn_uniform_k3(A,p,'tol',1e-11,'scalarTol',1e-14,'verbose',false);
assert(b.converged && max(abs(b.B-expected),[],'all') < 1e-9);
endpoint = broadcast_sinkhorn_nonuniform_k23(zeros(n,n),A,p,0,1, ...
    'tol',1e-11,'scalarTol',1e-14,'verbose',false,'doMixing',false);
assert(endpoint.converged && max(abs(endpoint.B3-expected),[],'all') < 1e-9);
badA = A; badA(1,5,6) = 0;
must_error(@() broadcast_sinkhorn_uniform_k3(badA,p,'verbose',false));
must_error(@() broadcast_sinkhorn_uniform_k3(zeros(3,3,3),ones(3,1)/3,'verbose',false));
must_error(@() merge_sinkhorn_nonuniform_k23(ones(3)/3,ones(3,3,3)/3,ones(3,1)/3,0,1));
fprintf('All small-problem solver regression tests passed.\n');
end

function must_error(fun)
caught = false;
try
    fun();
catch
    caught = true;
end
assert(caught,'Expected invalid input to be rejected.');
end
