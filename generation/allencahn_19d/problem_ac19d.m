function P = problem_ac19d()
%PROBLEM_AC19D  Specification of the 19-D Allen-Cahn optimal-control problem.
%   Single source of truth for the generation stage. The state is the vector
%   of interior nodes of a 1-D spatial grid (N_grid = 20 -> d = 19). The
%   value function is approximated at the initial time t0 (purely spatial:
%   no time derivatives are produced for this experiment).
%
%   The dynamics, running cost and terminal cost are baked into the solver
%   functions (bvpfcn / bcfcn), so no terminal-cost handles are needed here.
%
%   Fields:
%     P.name        problem identifier
%     P.d           state dimension (= 19)
%     P.lb, P.ub    sampling box   ([-1,1]^19)
%     P.n_hess      stored Hessian entries = d(d+1)/2 (= 190)
%     P.hess_order  Hessian vectorisation order
%     P.t0, P.T     initial / terminal time
%     P.beta, P.nu  control weight / diffusion coefficient
%     P.N_grid      spatial grid size

    d = 19;
    P.name       = 'allencahn_19d';
    P.d          = d;
    P.lb         = -ones(1, d);
    P.ub         =  ones(1, d);
    P.n_hess     = d*(d+1)/2;        % = 190
    P.hess_order = 'diagonal i=1..d first, then off-diagonal i<j (row-major)';

    P.t0     = 0.0;
    P.T      = 4.0;
    P.beta   = 0.01;
    P.nu     = 0.1;
    P.N_grid = 20;
end
