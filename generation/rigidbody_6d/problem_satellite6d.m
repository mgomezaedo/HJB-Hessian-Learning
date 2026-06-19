function P = problem_satellite6d()
%PROBLEM_SATELLITE6D  Specification of the 6-D rigid-body satellite OCP.
%   Single source of truth shared by the generation stage. State is
%   y = (phi, theta, psi, w1, w2, w3): three Euler angles + three body rates.
%   Finite-horizon problem on [t0, T] with quadratic running and terminal
%   cost; data generated at the initial time t0 (approximates V(t0, x)).
%
%   Fields:
%     P.name        problem identifier
%     P.d           state dimension (= 6)
%     P.lb, P.ub    sampling box  ([-pi/3,pi/3]^3 angles, [-pi/4,pi/4]^3 rates)
%     P.n_hess      stored Hessian entries = d(d+1)/2 (= 21)
%     P.hess_order  Hessian vectorisation order
%     P.t0, P.T     initial / terminal time
%     P.beta        control weight (= 1/4, matches satellite_pmp_rhs)
%     P.psi_val/grad/hess  terminal-cost handles  Psi(x) = alpha3*||x||^2

    d = 6;
    box = [pi/3 pi/3 pi/3 pi/4 pi/4 pi/4];

    P.name       = 'rigidbody_6d';
    P.d          = d;
    P.lb         = -box;
    P.ub         =  box;
    P.n_hess     = d*(d+1)/2;        % = 21
    P.hess_order = 'diagonal i=1..d first, then off-diagonal i<j';

    P.t0   = 0.0;
    P.T    = 20.0;
    P.beta = 0.25;                   % must match the hardcoded value in the engine

    alpha3 = 0.5;                    % terminal-cost weight
    P.psi_val  = @(x) alpha3 * (x.'*x);
    P.psi_grad = @(x) 2*alpha3 * x;
    P.psi_hess = @(x) 2*alpha3 * eye(d);
end
