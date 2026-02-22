function y = merge_map_k3(Xi, q)
%MERGE_MAP_K3  Apply the merging polynomial stochastic operator for k=3.
%   y(j) = sum_{i1,i2} Xi(i1,i2,j) q(i1) q(i2)

    n = size(Xi,1);
    q = q(:);

    y = zeros(n,1);
    for j = 1:n
        s = 0;
        for i1 = 1:n
            qi1 = q(i1);
            for i2 = 1:n
                s = s + Xi(i1,i2,j) * qi1 * q(i2);
            end
        end
        y(j) = s;
    end
end