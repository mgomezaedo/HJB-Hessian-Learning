function D = gen_dataset_rb6d(N, seed)
%GEN_DATASET_RB6D  Generate one labelled dataset for the 6-D satellite problem.
%   D = GEN_DATASET_RB6D(N, seed) draws N quasi-random (scrambled Sobol)
%   points on the domain, solves the PMP + Riccati system at each point via
%   solve_pmp_riccati_par (PARALLEL over points, parfor), keeps the points
%   with reliable data, and assembles the STANDARD dataset schema:
%       D.X    (M x 6)    sample points (M <= N after discarding failures)
%       D.V    (M x 1)    value      V(t0, x)
%       D.G    (M x 6)    gradient   nabla_x V
%       D.H    (M x 21)   Hessian, upper triangle [diag; off-diag i<j]
%       D.dtV  (M x 1)    partial_t V   (kept for the time-dependent stage)
%       D.meta struct     problem metadata + seed + kept count
%
%   Requires the Statistics and Machine Learning Toolbox (sobolset).
%   Parallel speedup needs the Parallel Computing Toolbox; without it,
%   parfor inside the engine runs serially (still correct).
%
%   NOTE: partial_tt V is not computed in 6D (engine stores NaN), so it is
%   not included here. The 6-D BVPs over the long horizon are expensive and
%   some fail (conjugate points), so M is typically noticeably below N.

    P = problem_satellite6d();
    d = P.d;

    % --- scrambled Sobol points on the box (reproducible via seed) ---
    rng(seed);
    S = sobolset(d);
    S = scramble(S, 'MatousekAffineOwen');
    U = net(S, N);                          % N x d in [0,1]^d
    Xbox = P.lb + (P.ub - P.lb) .* U;       % N x d
    X0 = Xbox.';                            % d x N (engine wants columns)

    % --- solve PMP + Riccati for all points (parallel engine) ---
    [Dataset, flags] = solve_pmp_riccati_par(P.t0, X0, P.T, ...
        P.psi_val, P.psi_grad, P.psi_hess, P.beta, true);

    % --- keep only points with reliable value AND Hessian ---
    keep = [flags.has_H];
    Dk = Dataset(:, keep);
    M  = size(Dk, 2);

    % --- assemble standard schema ---
    %   rows: 1:d x0 | d+1 V | d+2:2d+1 gradV | 2d+2 dtV
    %         2d+3:2d+2+d^2 Hessian (col-major d x d) | end dttV (NaN in 6D)
    D.X   = Dk(1:d, :).';                    % M x 6
    D.V   = Dk(d+1, :).';                    % M x 1
    D.G   = Dk(d+2:2*d+1, :).';              % M x 6
    D.dtV = Dk(2*d+2, :).';                  % M x 1

    Hcols = Dk(2*d+3 : 2*d+2+d^2, :);        % d^2 x M (col-major Hessians)
    Hv = zeros(M, d*(d+1)/2);
    for k = 1:M
        Hv(k, :) = hess_full_to_vec(Hcols(:, k), d);
    end
    D.H = Hv;                                % M x 21 -> [diag; off-diag i<j]

    D.meta = struct('problem', P.name, 'd', d, 'lb', P.lb, 'ub', P.ub, ...
                    'n_hess', P.n_hess, 'seed', seed, ...
                    'sampling', 'scrambled Sobol', 'hess_order', P.hess_order, ...
                    't0', P.t0, 'T', P.T, 'beta', P.beta, ...
                    'n_requested', N, 'n_kept', M);

    fprintf('gen_dataset_rb6d: kept %d / %d points (seed %d)\n', M, N, seed);
end
