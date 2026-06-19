function P = problem_vdp2d()
%PROBLEM_VDP2D  Specification of the 2-D Van der Pol optimal-control problem.
%   Single source of truth shared by the generation stage. Defines the
%   domain, the horizon, the control weight and the terminal cost used to
%   produce the value-function data via the PMP + Riccati solver.
%
%   Finite-horizon problem on [t0, T] with running cost x1^2 + x2^2,
%   control weight beta, and zero terminal cost. The data is generated at
%   the initial time t0 (i.e. it approximates V(t0, x) over the domain).
%
%   Fields:
%     P.name        problem identifier
%     P.d           state dimension (= 2)
%     P.lb, P.ub    sampling box for x   ([-3,3]^2)
%     P.n_hess      stored Hessian entries = d(d+1)/2 (= 3)
%     P.hess_order  Hessian vectorisation order
%     P.t0, P.T     initial / terminal time
%     P.beta        control weight
%     P.psi_val/grad/hess  terminal-cost handles (zero terminal cost here)

    d = 2;
    P.name       = 'vanderpol_2d';
    P.d          = d;
    P.lb         = [-3, -3];
    P.ub         = [ 3,  3];
    P.n_hess     = d*(d+1)/2;        % = 3
    P.hess_order = '[h11, h22, h12] (diagonal first, then off-diagonal)';

    P.t0   = 0.0;
    P.T    = 3.0;
    P.beta = 0.1;

    % zero terminal cost  Psi(x) = 0
    P.psi_val  = @(x) 0;
    P.psi_grad = @(x) zeros(2,1);
    P.psi_hess = @(x) zeros(2,2);
end
