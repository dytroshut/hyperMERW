function y = merge_map_k3(M3,q)
%MERGE_MAP_K3 Apply a uniform k=3 merging transition tensor.
%
%   y(j) = sum_{i1,i2} M3(i1,i2,j) q(i1) q(i2).

    validateattributes(M3, {'numeric'}, {'real','finite','nonnegative'});
    if ndims(M3) ~= 3
        error('M3 must be an n-by-n-by-n tensor.');
    end
    n = size(M3,1);
    if size(M3,2) ~= n || size(M3,3) ~= n
        error('M3 must be n-by-n-by-n.');
    end

    q = q(:);
    validateattributes(q, {'numeric'}, {'real','finite','nonnegative','numel',n});

    y = zeros(n,1);
    qq = q*q.';
    for j = 1:n
        y(j) = sum(M3(:,:,j).*qq,'all');
    end
end
