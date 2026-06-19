function D = gen_dataset_vdp2d(N, seed)
%GEN_DATASET_VDP2D  Generate one labelled dataset for the 2-D Van der Pol problem.
%   D = GEN_DATASET_VDP2D(N, seed) draws N quasi-random (scrambled Sobol)
%   points on the domain, solves the PMP + Riccati system at each point via
%   solve_pmp_riccati, keeps the points with reliable data, and assembles
%   the STANDARD dataset schema (same as every other experiment):
%       D.X    (M x 2)   sample points (M <= N after discarding failures)
%       D.V    (M x 1)   value      V(t0, x)
%       D.G    (M x 2)   gradient   nabla_x V
%       D.H    (M x 3)   Hessian    [h11, h22, h12]
%       D.dtV  (M x 1)   partial_t V    (kept for the time-dependent stage)
%       D.dttV (M x 1)   partial_tt V   (kept for the time-dependent stage)
%       D.meta struct    problem metadata + seed + how many points kept
%
%   Requires the Statistics and Machine Learning Toolbox (sobolset).
%   Some boundary-value solves fail (conjugate points, etc.); those points
%   are discarded, so M is typically a bit smaller than N.

    P = problem_vdp2d();
    d = P.d;

    % --- scrambled Sobol points on the box (reproducible via seed) ---
    rng(seed);
    S = sobolset(d);
    S = scramble(S, 'MatousekAffineOwen');
    U = net(S, N);                          % N x d in [0,1]^d
    Xbox = P.lb + (P.ub - P.lb) .* U;       % N x d
    X0 = Xbox.';                            % 2 x N (solver wants columns)

    % --- solve PMP + Riccati for all points ---
    [Dataset, flags] = solve_pmp_riccati(P.t0, X0, P.T, ...
        P.psi_val, P.psi_grad, P.psi_hess, P.beta, true);

    % --- keep only points with reliable value AND Hessian ---
    keep = [flags.has_H];                   % logical 1 x N
    Dk = Dataset(:, keep);
    M  = size(Dk, 2);

    % --- assemble standard schema ---
    %   Dataset rows: 1-2 x0 | 3 V | 4-5 gradV | 6 dtV
    %                 7-10 Riccati P0 col-major [P11;P21;P12;P22] | 11 dttV
    D.X    = Dk(1:2, :).';                       % M x 2
    D.V    = Dk(3, :).';                         % M x 1
    D.G    = Dk(4:5, :).';                       % M x 2
    D.H    = [Dk(7,:); Dk(10,:); Dk(9,:)].';     % M x 3  -> [h11, h22, h12]
    D.dtV  = Dk(6, :).';                         % M x 1
    D.dttV = Dk(11, :).';                        % M x 1

    D.meta = struct('problem', P.name, 'd', d, 'lb', P.lb, 'ub', P.ub, ...
                    'n_hess', P.n_hess, 'seed', seed, ...
                    'sampling', 'scrambled Sobol', 'hess_order', P.hess_order, ...
                    't0', P.t0, 'T', P.T, 'beta', P.beta, ...
                    'n_requested', N, 'n_kept', M);

    fprintf('gen_dataset_vdp2d: kept %d / %d points (seed %d)\n', M, N, seed);
end
