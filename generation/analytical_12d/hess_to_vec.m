function hvec = hess_to_vec(H, d)
%HESS_TO_VEC  Vectorise the upper triangle of a Hessian matrix.
%   hvec = HESS_TO_VEC(H, d) returns a 1 x d(d+1)/2 row vector with the
%   d diagonal entries first (i = 1..d), followed by the off-diagonal
%   entries with i < j scanned row by row.
%
%   IMPORTANT: this ordering MUST match the second-derivative basis
%   ordering produced by hc_basis on the regression side. If you change
%   one, change the other.
    n_hess = d*(d+1)/2;
    hvec = zeros(1, n_hess);
    idx = 1;
    for i = 1:d                 % diagonal
        hvec(idx) = H(i,i);
        idx = idx + 1;
    end
    for i = 1:d                 % off-diagonal, i < j
        for j = i+1:d
            hvec(idx) = H(i,j);
            idx = idx + 1;
        end
    end
end
