function [Dataset, flags] = solve_pmp_riccati(t0, X0, t_end, psi_val, psi_grad, psi_hess, beta, compute_riccati)
% SOLVE_PMP_RICCATI  Generate PMP+Riccati dataset for N initial conditions.
%
% For each column x0 of X0, solves the Pontryagin BVP over [t0, t_end] for
% the Van der Pol optimal control problem (see vdp_cost--vdp_state in paper),
% and optionally integrates the matrix Riccati equation backward along the
% optimal trajectory to obtain the spatial Hessian of V at (t0, x0).
%
% Inputs:
%   t0              : initial time (scalar), must satisfy t0 < t_end
%   X0              : (2 x N) matrix of initial states
%   t_end           : terminal time of slab (scalar)
%   psi_val         : @(x)->scalar   terminal cost value
%   psi_grad        : @(x)->2x1      terminal cost gradient
%   psi_hess        : @(x)->2x2      terminal cost Hessian
%   beta            : control weight (scalar > 0)
%   compute_riccati : (optional, default true)
%                     if false, skips Riccati integration entirely
%
% Outputs:
%   Dataset : (11 x N) matrix with rows:
%     1-2  : x0
%     3    : V(t0, x0)
%     4-5  : nabla_x V(t0, x0) = adjoint p(t0)
%     6    : partial_t V(t0, x0)  (from HJB identity)
%     7-10 : Riccati solution P0 stored col-major: [P(1,1); P(2,1); P(1,2); P(2,2)]
%            where P0 = nabla^2_x V(t0, x0)  (sign convention: R = nabla^2_x V)
%            NOTE: assemble_regression reorders these to [h11, h22, h12].
%     11   : partial_tt V(t0, x0)
%
%   flags : struct array (1 x N) with fields:
%     .ftype   : integer code (see below)
%     .has_V   : logical - V / nabla_x V / partial_t V are available
%     .clean_V : logical - above are reliable (BVP residual <= BVP_RESID_NOISY)
%     .has_H   : logical - P0 / partial_tt V are available
%     .clean_H : logical - above are reliable (clean_V AND Riccati ok)
%
% ftype codes (only reachable values):
%   0  : BVP clean, Riccati ok              - all data clean
%   1  : BVP failed or intolerable          - nothing usable
%   2  : BVP clean, Riccati blowup          - V/grad/dt_V clean only
%   3  : BVP clean, Riccati ODE crash       - V/grad/dt_V clean only
%   4  : BVP noisy (residual in (NOISY,MAX]), Riccati ok  - all data, noisy
%   5  : BVP noisy, Riccati blowup          - V/grad/dt_V noisy only
%   6  : BVP noisy, Riccati crash           - V/grad/dt_V noisy only
%   10 : compute_riccati=false, BVP clean or noisy  - V/grad/dt_V only
%
% Residual thresholds:
%   BVP_RESID_NOISY = 1e-2  : above -> noisy; at or below -> clean
%   BVP_RESID_MAX   = 1.0   : above -> intolerable (has_V = false, ftype = 1)

if nargin < 8
    compute_riccati = true;
end

N       = size(X0, 2);
Dataset = NaN(11, N);

default_flag = struct('ftype',1,'has_V',false,'clean_V',false,'has_H',false,'clean_H',false);
flags = repmat(default_flag, 1, N);

BVP_RESID_NOISY = 1e-2;
BVP_RESID_MAX   = 1.0;

% bvp5c options built once outside the loop (analytic Jacobians provided).
% N_mesh fixed at 201 regardless of slab length to guarantee sufficient
% mesh density at the required tolerances.
N_mesh   = max(201, round(301 * (t_end - t0) / 3));
xmesh    = linspace(t0, t_end, N_mesh);
bvp_opts = bvpset('FJacobian',  @(t,y) pontryagin_jac(t, y, beta), ...
                  'BCJacobian', @(ya,yb) bc_jac(ya, yb, psi_hess), ...
                  'AbsTol', 1e-10, 'RelTol', 1e-9, 'NMax', 500000);

for i = 1:N
    x0 = X0(:, i);

    % ----------------------------------------------------------------
    % 1. Warm-start guesses for the initial adjoint p(t0).
    %    First guess:  psi_grad(x0) (terminal gradient of previous slab
    %                  is a reasonable proxy for p(t0)).
    %    Second guess: negated, to handle sign ambiguity.
    %    Fallback to [1;1] when psi_grad(x0) is zero (e.g. last slab).
    % ----------------------------------------------------------------
    try
        p_ws0 = psi_grad(x0);
    catch
        p_ws0 = zeros(2,1);
    end

    if norm(p_ws0) < 1e-10
        p_ws0 = [1;1];
    end
    p_ws0 = p_ws0 / max(1, norm(p_ws0)/10);

    % ----------------------------------------------------------------
    % 2. Solve BVP (up to 2 initial-guess attempts)
    % ----------------------------------------------------------------
    [sol, bvp_ok, bvp_clean, bvp_resid] = ...
        solve_bvp_with_retry(x0, xmesh, beta, p_ws0, psi_grad, bvp_opts, ...
                             BVP_RESID_NOISY, BVP_RESID_MAX);

    if ~bvp_ok
        flags(i).ftype = 1;
        continue
    end

    % ----------------------------------------------------------------
    % 3. Value, gradient, and partial_t V  (always computed if BVP ok)
    %
    % V(t0,x0) = integral of running cost along optimal trajectory + Psi(y(T)).
    % nabla_x V(t0,x0) = p(t0)  [PMP / DPP relation].
    % partial_t V from HJB:
    %   partial_t V = -l(y0) - p0'*f(y0) + (1/4beta)*p2^2
    % where l(y)=y1^2+y2^2, f is VDP drift, g=[0;1] so (g*g'*p)=[0;p2].
    % ----------------------------------------------------------------
    running = sol.y(1,:).^2 + sol.y(2,:).^2 + (1/(4*beta))*sol.y(4,:).^2;
    V_val   = trapz(sol.x, running) + psi_val([sol.y(1,end); sol.y(2,end)]);

    y0   = [sol.y(1,1); sol.y(2,1)];
    p0   = [sol.y(3,1); sol.y(4,1)];
    f0   = vdp_drift(y0);
    dt_V = -(y0(1)^2 + y0(2)^2) - p0'*f0 + (1/(4*beta))*(p0(2)^2);

    % ----------------------------------------------------------------
    % 4. Riccati integration (optional)
    % ----------------------------------------------------------------
    [ric_ok, ric_blowup, P_sol] = ...
        solve_riccati_backward(sol, t0, t_end, beta, psi_hess, compute_riccati);

    % ----------------------------------------------------------------
    % 5. Classify quality and assign flags
    % ----------------------------------------------------------------
    [ft, has_V, clean_V, has_H, clean_H] = ...
        classify_flags(bvp_clean, bvp_resid, ric_ok, ric_blowup, ...
                       compute_riccati, BVP_RESID_MAX);

    flags(i) = struct('ftype',ft,'has_V',has_V,'clean_V',clean_V, ...
                      'has_H',has_H,'clean_H',clean_H);

    % ----------------------------------------------------------------
    % 6. Store data
    % ----------------------------------------------------------------
    Dataset(1:2, i) = x0;

    if has_V
        Dataset(3,   i) = V_val;
        Dataset(4:5, i) = p0;
        Dataset(6,   i) = dt_V;
    end

    if has_H
        % P0 is the Riccati solution at t0: P0 = nabla^2_x V(t0, x0).
        % Symmetrize to suppress numerical drift from ODE integration.
        P0    = reshape(P_sol(end,:)', 2, 2);
        P0    = (P0 + P0') / 2;
        g0    = [0; 1];
        fy0   = vdp_drift_jac(y0);
        dtt_V = compute_dtt_V(y0, p0, P0, f0, g0, fy0, beta);

        Dataset(7:10, i) = P0(:);   % col-major: [P11; P21; P12; P22]
        Dataset(11,   i) = dtt_V;
    end

end  % i
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [sol, bvp_ok, bvp_clean, bvp_resid] = ...
        solve_bvp_with_retry(x0, xmesh, beta, p_ws0, psi_grad, opts, ...
                             BVP_RESID_NOISY, BVP_RESID_MAX)
% SOLVE_BVP_WITH_RETRY  Attempt Pontryagin BVP solve with two guess directions.
%
% Tries p_ws0 (iguess=1), then -p_ws0 (iguess=2) as initial adjoint guesses.
% Accepts the first solution with residual <= BVP_RESID_MAX.
%
% Outputs:
%   sol       : bvp5c solution struct (meaningful only if bvp_ok=true)
%   bvp_ok    : true if a tolerable solution was found
%   bvp_clean : true if accepted residual <= BVP_RESID_NOISY
%   bvp_resid : maximum residual of accepted solution (Inf if bvp_ok=false)

sol       = [];
bvp_ok    = false;
bvp_clean = false;
bvp_resid = Inf;

for iguess = 1:2
    p_guess = (3 - 2*iguess) * p_ws0;   % +p_ws0 for iguess=1, -p_ws0 for iguess=2
    solinit = bvpinit(xmesh, [x0; p_guess]);

    try
        lastwarn('');
        wstate = warning('off','all');

        output_str = evalc(['sol_try = bvp5c(@(t,y) pontryagin_rhs(t, y, beta), ' ...
            '@(ya,yb) bc_func(ya, yb, x0, psi_grad), solinit, opts);']);

        warning(wstate);

    catch
        warning(wstate);
        continue
    end

    % Parse residual from bvp5c output string.
    % bvp5c prints "The maximum residual is X.XXe-YY" on convergence.
    % Silent convergence (no string, no warning) is treated as residual=0.
    % A suppressed warning without the residual string signals an unreliable
    % solve and is treated as residual=Inf (noisy).
    tok = regexp(output_str, 'maximum residual is ([0-9eE.+-]+)', 'tokens');

    if isempty(tok)
        [wmsg, ~] = lastwarn;
        if isempty(wmsg)
            bvp_resid_try = 0;
        else
            bvp_resid_try = Inf;
        end
    else
        residuals     = cellfun(@(t) str2double(t{1}), tok);
        bvp_resid_try = max(residuals);
    end

    if bvp_resid_try <= BVP_RESID_MAX
        sol       = sol_try;
        bvp_ok    = true;
        bvp_clean = (bvp_resid_try <= BVP_RESID_NOISY);
        bvp_resid = bvp_resid_try;
        break
    end

end  % iguess
end

% -------------------------------------------------------------------------

function [ric_ok, ric_blowup, P_sol] = ...
        solve_riccati_backward(sol, t0, t_end, beta, psi_hess, compute_riccati)
% SOLVE_RICCATI_BACKWARD  Integrate matrix Riccati ODE backward along trajectory.
%
% Integrates dP/dt = -Hyp*P - P*Hpy - P*Hpp*P - Hyy from t_end to t0
% using the optimal trajectory stored in the bvp5c solution struct sol.
% P(t) = nabla^2_x V(t, y*(t))  (sign convention R = nabla^2_x V).
%
% If compute_riccati=false, returns immediately with all outputs empty/false.
%
% Outputs:
%   ric_ok     : true if integration reached t0 without blowup or crash
%   ric_blowup : true if ||P|| exceeded 1e6 (conjugate point detected)
%   P_sol      : ode45 solution matrix (rows = time steps); valid if ric_ok

ric_ok     = false;
ric_blowup = false;
P_sol      = [];

if ~compute_riccati
    return
end

y_end      = [sol.y(1,end); sol.y(2,end)];
P_terminal = psi_hess(y_end);

y1_i = griddedInterpolant(sol.x, sol.y(1,:), 'pchip');
y2_i = griddedInterpolant(sol.x, sol.y(2,:), 'pchip');
p1_i = griddedInterpolant(sol.x, sol.y(3,:), 'pchip');
p2_i = griddedInterpolant(sol.x, sol.y(4,:), 'pchip');
trajs = {y1_i, y2_i, p1_i, p2_i};

opts_ode = odeset('AbsTol', 1e-8, 'RelTol', 1e-7, ...
                  'Events', @riccati_blowup_event);
try
    [~, P_sol, te, ~, ~] = ode45(@(t,P) riccati_rhs(t, P, trajs, beta), ...
                                  [t_end, t0], P_terminal(:), opts_ode);
    if isempty(te)
        ric_ok = true;
    else
        ric_blowup = true;   % conjugate point: ||P|| exceeded 1e6
    end
catch
    % ric_ok and ric_blowup both remain false -> Riccati ODE crash
end
end

% -------------------------------------------------------------------------

function [ft, has_V, clean_V, has_H, clean_H] = ...
        classify_flags(bvp_clean, bvp_resid, ric_ok, ric_blowup, ...
                       compute_riccati, BVP_RESID_MAX)
% CLASSIFY_FLAGS  Map BVP and Riccati outcomes to ftype and availability flags.
%
% Called only for points where bvp_ok=true (failed BVPs are assigned ftype=1
% upstream and skipped via continue before reaching this function).
%
% Reachable ftype values:
%   0  : BVP clean, Riccati ok
%   1  : BVP intolerable (residual > BVP_RESID_MAX)  - nothing usable
%   2  : BVP clean, Riccati blowup
%   3  : BVP clean, Riccati crash
%   4  : BVP noisy, Riccati ok
%   5  : BVP noisy, Riccati blowup
%   6  : BVP noisy, Riccati crash
%   10 : compute_riccati=false, BVP clean or noisy

intolerable = ~bvp_clean && (bvp_resid > BVP_RESID_MAX);

if intolerable
    ft      = 1;
    has_V   = false;  clean_V = false;
    has_H   = false;  clean_H = false;
    return
end

% BVP is tolerable (clean or noisy) from here on.
has_V   = true;
clean_V = bvp_clean;

if ~compute_riccati
    ft      = 10;
    has_H   = false;  clean_H = false;
    return
end

if bvp_clean && ric_ok
    ft = 0;
elseif bvp_clean && ric_blowup
    ft = 2;
elseif bvp_clean       % ~ric_ok && ~ric_blowup: Riccati ODE crash
    ft = 3;
elseif ric_ok          % noisy BVP, Riccati ok
    ft = 4;
elseif ric_blowup      % noisy BVP, Riccati blowup
    ft = 5;
else                   % noisy BVP, Riccati crash
    ft = 6;
end

has_H   = ric_ok;
clean_H = bvp_clean && ric_ok;
end

% =========================================================================
% Pontryagin system for VDP (adjoint equations + optimal control)
%   state vector: y = [y1; y2; p1; p2]
%   optimal control: u* = -(1/2beta)*p2
% =========================================================================
function dydt = pontryagin_rhs(~, y, beta)
dydt = [ y(2);
         y(2) - y(1) - y(2)*y(1)^2 - (1/(2*beta))*y(4);
         (1 + 2*y(1)*y(2))*y(4) - 2*y(1);
        -y(3) - (1 - y(1)^2)*y(4) - 2*y(2) ];
end

function J = pontryagin_jac(~, y, beta)
J      = zeros(4,4);
J(1,:) = [0,               1,             0,  0            ];
J(2,:) = [-1-2*y(1)*y(2),  1-y(1)^2,      0,  -1/(2*beta)  ];
J(3,:) = [2*y(2)*y(4)-2,   2*y(1)*y(4),   0,  1+2*y(1)*y(2)];
J(4,:) = [2*y(1)*y(4),    -2,            -1,  -(1-y(1)^2)  ];
end

% =========================================================================
% Boundary conditions: y(t0) = x0,  p(T) = nabla Psi(y(T))
% =========================================================================
function res = bc_func(ya, yb, x0, psi_grad)
p_term = psi_grad([yb(1); yb(2)]);
res = [ ya(1)-x0(1); ya(2)-x0(2);
        yb(3)-p_term(1); yb(4)-p_term(2) ];
end

function [dBCdya, dBCdyb] = bc_jac(ya, yb, psi_hess) %#ok<INUSL>
% ya is required by bvp5c's BCJacobian interface but is unused: the initial
% BC ya(1:2)=x0 does not depend on ya(3:4) or on yb.
H = psi_hess([yb(1); yb(2)]);
dBCdya = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0];
dBCdyb = [0 0 0 0; 0 0 0 0;
         -H(1,1) -H(1,2) 1 0;
         -H(2,1) -H(2,2) 0 1];
end

% =========================================================================
% Riccati ODE: dP/dt = -Hyp*P - P*Hpy - P*Hpp*P - Hyy  (riccati_spatial)
% Integrated backward from t_end to t0 along the optimal trajectory.
% P(t) = nabla^2_x V(t, y*(t)).
% Coefficient matrices Hpy, Hyp, Hpp, Hyy as in vdp_riccati_coeffs.
% =========================================================================
function dPdt = riccati_rhs(t, P_vec, trajs, beta)
y1 = trajs{1}(t);  y2 = trajs{2}(t);  p2 = trajs{4}(t);
P  = reshape(P_vec, 2, 2);

Hpy = [0,           1;
       -2*y1*y2-1,  1-y1^2];
Hyp = Hpy';
Hpp = [0,  0;
       0,  -1/(2*beta)];
Hyy = [2-2*p2*y2,  -2*p2*y1;
       -2*p2*y1,    2       ];

dPdt = (-Hyp*P - P*Hpy - P*Hpp*P - Hyy);
dPdt = dPdt(:);
end

function [value, isterminal, direction] = riccati_blowup_event(~, P_vec)
% Terminate Riccati integration when ||P|| > 1e6 (conjugate point).
value      = 1e6 - norm(P_vec);
isterminal = 1;
direction  = -1;
end

% =========================================================================
% VDP dynamics and Jacobian
% =========================================================================
function f = vdp_drift(y)
% Unforced VDP drift: f(y) = [y2; -y1 + y2*(1-y1^2)]  (u=0 in vdp_state)
f = [y(2); y(2)-y(1)-y(2)*y(1)^2];
end

function Jf = vdp_drift_jac(y)
Jf = [0,              1;
      -1-2*y(1)*y(2), 1-y(1)^2];
end

% =========================================================================
% Second temporal derivative of V at (t0, x0).
%
% Differentiating the HJB equation with respect to t and evaluating at
% (t0, x0) along the optimal trajectory gives:
%   partial_tt V = -(nabla_x partial_t V)' * v0
% where v0 = f(y0) - (1/2beta)*g*g'*p0 is the closed-loop velocity, and
%   nabla_x partial_t V = 2*y0 + P0*f0 + Jf(y0)'*p0 - (1/4beta)*grad_Q
% with grad_Q = 2*P0*(g*g'*p0)  [gradient of (1/4beta)*p'*g*g'*p w.r.t. x,
% positive sign].
%
% Inputs:
%   y0, p0 : state and adjoint at t0
%   P0     : Riccati solution P0 = nabla^2_x V(t0, x0) 
%   f0     : vdp_drift(y0)
%   g0     : control input column [0; 1]
%   fy0    : vdp_drift_jac(y0)
%   beta   : control weight
% =========================================================================
function dtt = compute_dtt_V(y0, p0, P0, f0, g0, fy0, beta)
gg      = g0*g0';
v0      = f0 - (1/(2*beta))*gg*p0;
grad_Q  = 2*P0*(gg*p0);                              % positive sign
bracket = 2*y0 + P0*f0 + fy0'*p0 - (1/(4*beta))*grad_Q;
dtt     = bracket'*v0;
end