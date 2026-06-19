function hvec = hess_full_to_vec(Pvec, d)
%HESS_FULL_TO_VEC  Full (col-major) Hessian -> upper-triangular vector.
%   hvec = HESS_FULL_TO_VEC(Pvec, d) takes the d^2 column-major entries of a
%   d-by-d Hessian (as stored by the PMP+Riccati engine) and returns a
%   1 x d(d+1)/2 row vector with the d diagonal entries first (i = 1..d),
%   then the off-diagonal entries with i < j.
%
%   This ordering MUST match the second-derivative basis ordering produced
%   by hc_basis on the regression side.

    P = reshape(Pvec, d, d);
    P = (P + P.') / 2;              % symmetrise (suppress ODE drift)

    hvec = zeros(1, d*(d+1)/2);
    idx = 1;
    for i = 1:d                     % diagonal
        hvec(idx) = P(i,i); idx = idx + 1;
    end
    for i = 1:d                     % off-diagonal, i < j
        for j = i+1:d
            hvec(idx) = P(i,j); idx = idx + 1;
        end
    end
end
