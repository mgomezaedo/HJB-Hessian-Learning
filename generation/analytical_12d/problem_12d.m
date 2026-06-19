function P = problem_12d()
%PROBLEM_12D  Specification of the 12-D analytical benchmark (f_AS).
%   Single source of truth for the problem: dimension, domain, Hessian
%   ordering and the value/gradient/Hessian handles. BOTH the generation
%   and the regression stages read this, so nothing is hard-coded twice.
%
%   Fields:
%     P.name        problem identifier (used in filenames / metadata)
%     P.d           dimension (= 12)
%     P.lb, P.ub    lower/upper bounds of the hypercube domain (1 x d)
%     P.n_hess      number of stored Hessian entries = d(d+1)/2 (= 78)
%     P.hess_order  text describing the Hessian vectorisation order
%     P.value       handle: V = P.value(X),    X (N x d) -> V (N x 1)
%     P.grad        handle: G = P.grad(X),     X (N x d) -> G (N x d)
%     P.hess_vec    handle: H = P.hess_vec(X), X (N x d) -> H (N x n_hess)

    d = 12;
    P.name       = 'analytical_12d';
    P.d          = d;
    P.lb         = -ones(1, d);     % domain [-1, 1]^d
    P.ub         =  ones(1, d);
    P.n_hess     = d*(d+1)/2;       % = 78
    P.hess_order = 'diagonal i=1..d first, then off-diagonal i<j';

    % fun1dim12* expect points as COLUMNS (d x N); we work with rows (N x d).
    P.value    = @(X) fun1dim12(X.').';      % (N x d) -> (N x 1)
    P.grad     = @(X) fun1dim12x(X.').';     % (N x d) -> (N x d)
    P.hess_vec = @(X) eval_hess_vec(X, d);   % (N x d) -> (N x n_hess)
end

% -------------------------------------------------------------------------
function H = eval_hess_vec(X, d)
%EVAL_HESS_VEC  Vectorised Hessian for every row of X.
    N = size(X, 1);
    H = zeros(N, d*(d+1)/2);
    for j = 1:N
        Hf = fun1dim12xx(X(j,:).');   % d x d Hessian at point j
        H(j,:) = hess_to_vec(Hf, d);
    end
end
