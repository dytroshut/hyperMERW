% TEST_CORRECTED_MERGE
% Regression checks using the old saved synthetic references.

clear; clc;

scriptDirectory = fileparts(mfilename('fullpath'));
if isempty(scriptDirectory)
    scriptDirectory = pwd;
end
addpath(scriptDirectory);
oldFile = fullfile(scriptDirectory,'merge_mixing_curves_safe.mat');
if ~isfile(oldFile)
    error(['Copy this test and the corrected solver files into the original ' ...
        'project directory containing %s.'],oldFile);
end

old = load(oldFile,'A2','A3','Xi','p');

uniformFit = merge_sinkhorn_uniform_k3(old.A3,old.p, ...
    'maxIter',6000,'tol',1e-12,'rootTol',1e-12,'verbose',false);

assert(uniformFit.converged,'Corrected uniform solver did not converge.');
assert(max(uniformFit.finalRes) < 1e-10, ...
    'Corrected uniform residual is too large.');
assert(uniformFit.symmetryResidual < 1e-12, ...
    'Corrected uniform tensor is not back-symmetric.');
uniformReconstruction = zeros(size(uniformFit.M));
for j = 1:numel(uniformFit.p)
    uniformReconstruction(:,:,j) = uniformFit.K(:,:,j) ...
        .*uniformFit.U.*uniformFit.v(j).^uniformFit.omega;
end
assert(max(abs(uniformReconstruction-uniformFit.M),[],'all') < 1e-10, ...
    'Uniform K/U/v fields do not reconstruct the inferred tensor.');

oldObjective = kl_objective(old.Xi,old.A3);
newObjective = uniformFit.objective;
maxTensorChange = max(abs(uniformFit.Xi-old.Xi),[],'all');

fprintf('Uniform corrected solve\n');
fprintf('  old objective       = %.10f\n',oldObjective);
fprintf('  corrected objective = %.10f\n',newObjective);
fprintf('  maximum tensor change = %.6e\n',maxTensorChange);
fprintf('  output residual       = %.3e\n',uniformFit.rowResidual);
fprintf('  stationarity residual = %.3e\n',uniformFit.stationarityResidual);
fprintf('  contraction bound      = %.6f\n',uniformFit.contractionBound);
fprintf('  theorem certified      = %d\n',uniformFit.contractionCertified);

assert(newObjective <= oldObjective+1e-9, ...
    'Corrected objective should not exceed the old feasible objective.');

jointFit = merge_sinkhorn_nonuniform_k23( ...
    old.A2,old.A3,old.p,0.5,0.5, ...
    'maxIter',6000,'tol',1e-12,'rootTol',1e-12,'verbose',false);

assert(jointFit.converged,'Corrected joint solver did not converge.');
assert(max(jointFit.finalRes) < 1e-10, ...
    'Corrected joint residual is too large.');
assert(jointFit.symmetryResidual3 < 1e-12, ...
    'Corrected joint k=3 tensor is not back-symmetric.');
jointReconstruction2 = zeros(size(jointFit.M2));
jointReconstruction3 = zeros(size(jointFit.M3));
for j = 1:numel(jointFit.p)
    jointReconstruction2(:,j) = jointFit.K2(:,j) ...
        .*jointFit.U2.*jointFit.v(j).^jointFit.omega2;
    jointReconstruction3(:,:,j) = jointFit.K3(:,:,j) ...
        .*jointFit.U3.*jointFit.v(j).^jointFit.omega3;
end
assert(max(abs(jointReconstruction2-jointFit.M2),[],'all') < 1e-10, ...
    'Joint k=2 K/U/v fields do not reconstruct the inferred matrix.');
assert(max(abs(jointReconstruction3-jointFit.M3),[],'all') < 1e-10, ...
    'Joint k=3 K/U/v fields do not reconstruct the inferred tensor.');

fprintf('\nJoint corrected solve at (lambda2,lambda3)=(0.5,0.5)\n');
fprintf('  iterations             = %d\n',jointFit.iters);
fprintf('  M2 output residual     = %.3e\n',jointFit.rowResidual2);
fprintf('  M3 output residual     = %.3e\n',jointFit.rowResidual3);
fprintf('  stationarity residual  = %.3e\n',jointFit.stationarityResidual);
fprintf('  contraction bound      = %.6f\n',jointFit.contractionBound);
fprintf('  theorem certified      = %d\n',jointFit.contractionCertified);


function value = kl_objective(M,A)
    active = A > 0 & M > 0;
    value = sum(M(active).*log(M(active)./A(active)),'all');
end
