function write_readme(root)
%WRITE_README Generate the README and every displayed array from saved results.
if nargin < 1, root = fileparts(mfilename('fullpath')); end
b = load(fullfile(root,'results','broadcast_results_corrected.mat'),'results'); b = b.results;
u = load(fullfile(root,'results','broadcast_uniform_corrected.mat'),'uniform_results'); u = u.uniform_results;
m = load(fullfile(root,'results','merge_mixing_curves_corrected.mat'));
v = load(fullfile(root,'results','verification.mat'),'report'); v = v.report;
assert(v.passed,'Verify the results before publishing the README.');
fid = fopen(fullfile(root,'README.md'),'w');
assert(fid ~= -1,'Cannot create README.md.');
closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',fileread(fullfile(root,'documentation','README_intro.md')));
fprintf(fid,'\n## Reproduced results\n\n');
fprintf(fid,'MATLAB version used for the saved run: `%s`.\n\n',v.matlabVersion);
fprintf(fid,'### Broadcasting\n\n');
fprintf(fid,'| Weights (lambda2, lambda3) | Sweeps | Max row residual | Stationarity L1 | Spectral gap | L1 error at t=500 |\n');
fprintf(fid,'| --- | ---: | ---: | ---: | ---: | ---: |\n');
for q = 1:size(b.W,1)
    fprintf(fid,'| (%.1f, %.1f) | %d | %.3e | %.3e | %.9f | %.3e |\n', ...
        b.W(q,:),b.stats(q,1),v.broadcastRow(q),v.broadcastStationarity(q),b.stats(q,4),b.mixCurves(end,q));
end
fprintf(fid,'\nThe initial L1 error is %.12f for every broadcasting curve.\n',b.mixCurves(1,1));
fprintf(fid,'\n### Merging\n\n');
fprintf(fid,'| Weights (lambda2, lambda3) | Iterations | M2 max row residual | M3 max row residual | Stationarity L1 | Contraction bound | L1 error at t=50 |\n');
fprintf(fid,'| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for q = 1:size(m.W,1)
    if isnan(m.rowResidual2(q)), row2 = 'not active'; else, row2 = sprintf('%.3e',m.rowResidual2(q)); end
    fprintf(fid,'| (%.1f, %.1f) | %d | %s | %.3e | %.3e | %.6f | %.3e |\n', ...
        m.W(q,:),m.solverIterations(q),row2,m.rowResidual3(q),m.stationarityResidual(q),m.contractionBound(q),m.mixCurves(end,q));
end
fprintf(fid,'\nThe initial L1 error is %.12f for every merging curve. None of these bounds is below one.\n',m.initialError);
fprintf(fid,'\n### Verification\n\n');
fprintf(fid,'All saved-result checks passed. The largest checked log-scaling residual is %.3e for broadcasting and %.3e for merging.\n', ...
    max(v.broadcastKKT),max(v.mergeKKT));
fprintf(fid,'\n## Stationary distributions\n\n');
write_matrix(fid,'p_broadcast',b.p.',15);
write_matrix(fid,'p_merge',m.p.',15);
fprintf(fid,'\n## Main reported matrices\n\n');
fprintf(fid,'Uniform broadcasting projected kernel:\n\n');
write_matrix(fid,'P_uniform',u.Pproj,6);
q = find(all(abs(m.W-[0.5 0.5]) < 1e-12,2));
fprintf(fid,'\nMerging pairwise transition matrix at (lambda2,lambda3)=(0.5,0.5):\n\n');
write_matrix(fid,'M2',m.M2ByWeight{q},6);
fprintf(fid,'\n## Full adjacency/reference arrays\n\n');
fprintf(fid,'These are the normalized reference arrays used by the solvers. Their positive entries define the adjacency.\n\n');
open_details(fid,'Broadcasting: A2 and all eight A3 slices');
write_matrix(fid,'A2_broadcast',b.A2,6); write_tensor(fid,'A3_broadcast',b.A3);
close_details(fid);
open_details(fid,'Merging: A2 and all eight A3 slices');
write_matrix(fid,'A2_merge',m.A2,6); write_tensor(fid,'A3_merge',m.A3);
close_details(fid);
fprintf(fid,'\n## Full broadcasting transition tensors\n\n');
open_details(fid,'Uniform k=3: all eight B3 slices');
write_tensor(fid,'B3_uniform',u.B3); close_details(fid);
for q = 1:size(b.W,1)
    open_details(fid,sprintf('Weights (%.1f, %.1f): P, layer contributions, and every B3 slice',b.W(q,:)));
    write_matrix(fid,'P',b.Pproj_all{q},6);
    if b.W(q,1) > 0, write_matrix(fid,'B2_equals_P2',b.B2_all{q},6);
    else, fprintf(fid,'The k=2 layer is inactive. Its stored zero array is only bookkeeping.\n\n'); end
    write_matrix(fid,'P3',sum(b.B3_all{q},3),6);
    write_tensor(fid,'B3',b.B3_all{q});
    close_details(fid);
end
fprintf(fid,'\n## Full merging transition tensors\n\n');
for q = 1:size(m.W,1)
    open_details(fid,sprintf('Weights (%.1f, %.1f): M2 and every M3 receiver slice',m.W(q,:)));
    if isempty(m.M2ByWeight{q}), fprintf(fid,'Pure k=3 case. There is no inferred M2.\n\n');
    else, write_matrix(fid,'M2',m.M2ByWeight{q},6); end
    write_tensor(fid,'M3',m.M3ByWeight{q});
    close_details(fid);
end
fprintf('README generated from verified saved arrays.\n');
end

function write_tensor(fid,name,T)
for j = 1:size(T,3), write_matrix(fid,sprintf('%s(:,:,%d)',name,j),T(:,:,j),6); end
end

function write_matrix(fid,name,A,digits)
fprintf(fid,'```matlab\n%s = [\n',name);
format = sprintf(' %%.%df',digits);
for i = 1:size(A,1)
    fprintf(fid,'%s',sprintf(format,A(i,:)));
    fprintf(fid,';\n');
end
fprintf(fid,'];\n```\n\n');
end

function open_details(fid,title)
fprintf(fid,'<details>\n<summary>%s</summary>\n\n',title);
end

function close_details(fid)
fprintf(fid,'</details>\n\n');
end
