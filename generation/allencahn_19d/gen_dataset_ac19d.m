function D = gen_dataset_ac19d(N, seed)
%GEN_DATASET_AC19D  Generate one labelled dataset for the 19-D Allen-Cahn problem.
%   D = GEN_DATASET_AC19D(N, seed) draws N quasi-random (scrambled Sobol)
%   points on [-1,1]^19, solves the Pontryagin BVP and the Riccati equation
%   at each point (PARALLEL, parfor), keeps the successful points, and
%   assembles the STANDARD dataset schema:
%       D.X    (M x 19)    sample points (M <= N after discarding failures)
%       D.V    (M x 1)     value      V(t0, x)
%       D.G    (M x 19)    gradient   nabla_x V
%       D.H    (M x 190)   Hessian, upper triangle [diag; off-diag i<j]
%       D.meta struct      problem metadata + seed + kept count
%
%   Requires the Statistics and Machine Learning Toolbox (sobolset).
%   Parallel speedup needs the Parallel Computing Toolbox.
%
%   This mirrors the per-sample pipeline of the original generate_dataset_19d
%   (solvePontryagin -> computeValueFunction -> solveRiccati), but returns the
%   standard struct and the upper-triangular Hessian instead of the raw 400xN
%   pool. There are no time derivatives for this experiment.

    P  = problem_ac19d();
    d  = P.d;
    T  = P.T;
    d2 = d*d;

    % --- scrambled Sobol points on [-1,1]^d (reproducible via seed) ---
    rng(seed, 'twister');
    sob = scramble(sobolset(d, 'Skip', 1024), 'MatousekAffineOwen');
    u   = net(sob, N);                      % N x d in [0,1]^d
    Xs  = P.lb + (P.ub - P.lb) .* u;        % N x d

    % --- solve PMP + Riccati for all points (parallel) ---
    sizeofDataset = d + 1 + d + d2;         % 400
    DS = nan(sizeofDataset, N);

    parfor i = 1:N
        x = Xs(i, :);                       % 1 x d row
        try
            sol = solvePontryagin(x, T);
            V_i = computeValueFunction(sol);

            trajectories = cell(2*d, 1);
            for j = 1:d
                trajectories{j}   = griddedInterpolant(sol.x, sol.y(j, :));
                trajectories{j+d} = griddedInterpolant(sol.x, sol.y(j+d, :));
            end
            Pmat = solveRiccati(T, trajectories, d);

            col = zeros(sizeofDataset, 1);
            col(1:d) = x';
            col(d+1) = V_i;
            for j = 1:d,  col(d+1+j)   = sol.y(d+j, 1); end   % gradient = costate(0)
            for j = 1:d2, col(d+1+d+j) = Pmat(1, j);    end   % Hessian (flat)
            DS(:, i) = col;
        catch
            DS(:, i) = nan;
        end
    end

    % --- keep successful samples ---
    keep = ~any(isnan(DS), 1);
    DSk  = DS(:, keep);
    M    = size(DSk, 2);

    % --- assemble standard schema ---
    D.X = DSk(1:d, :).';                              % M x 19
    D.V = DSk(d+1, :).';                              % M x 1
    D.G = DSk(d+2:2*d+1, :).';                        % M x 19
    D.H = flat_to_uppertri(DSk(2*d+2:end, :), d);     % M x 190

    D.meta = struct('problem', P.name, 'd', d, 'lb', P.lb, 'ub', P.ub, ...
                    'n_hess', P.n_hess, 'seed', seed, ...
                    'sampling', 'scrambled Sobol', 'hess_order', P.hess_order, ...
                    't0', P.t0, 'T', P.T, 'beta', P.beta, 'nu', P.nu, ...
                    'n_requested', N, 'n_kept', M);

    fprintf('gen_dataset_ac19d: kept %d / %d points (seed %d)\n', M, N, seed);
end
