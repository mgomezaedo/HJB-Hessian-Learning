function D = gen_dataset_12d(N, seed)
%GEN_DATASET_12D  Generate one labelled dataset for the 12-D benchmark.
%   D = GEN_DATASET_12D(N, seed) draws N quasi-random (scrambled Sobol)
%   points on [-1,1]^12 and evaluates the exact value, gradient and
%   vectorised Hessian of f_AS at each point.
%
%   The scramble is seeded by rng(seed): same seed -> same points
%   (reproducible); different seeds -> different point sets (used for the
%   Monte-Carlo repetitions of the convergence study).
%
%   Requires the Statistics and Machine Learning Toolbox (sobolset).
%
%   Output struct D (standard schema, shared by every experiment):
%       D.X    (N x d)        sample points
%       D.V    (N x 1)        values
%       D.G    (N x d)        gradients
%       D.H    (N x n_hess)   Hessians, upper triangle (see hess_to_vec)
%       D.meta struct         problem metadata + seed

    P = problem_12d();
    d = P.d;

    % --- scrambled Sobol points (low-discrepancy, reproducible) ---
    rng(seed);
    S = sobolset(d);
    S = scramble(S, 'MatousekAffineOwen');
    U = net(S, N);                          % N x d in [0,1]^d
    X = P.lb + (P.ub - P.lb) .* U;          % map to [-1,1]^d

    D.X = X;
    D.V = P.value(X);
    D.G = P.grad(X);
    D.H = P.hess_vec(X);

    D.meta = struct('problem', P.name, 'd', d, 'lb', P.lb, 'ub', P.ub, ...
                    'n_hess', P.n_hess, 'seed', seed, ...
                    'sampling', 'scrambled Sobol', 'hess_order', P.hess_order);
end
