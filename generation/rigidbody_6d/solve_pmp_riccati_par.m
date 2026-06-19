function [Dataset, flags] = solve_pmp_riccati_par(t0, X0, t_end, psi_val, psi_grad, psi_hess, beta, compute_riccati)
% SOLVE_PMP_RICCATI_PAR  Parallel version — uses parfor over initial conditions.
%
% Same interface as solve_pmp_riccati but loops with parfor.
% Use this when calling from a sequential context (e.g., Pass 1 snapshots).
% Do NOT use inside another parfor (nested parfor not allowed).

if nargin < 8, compute_riccati = true; end

d = 6;
n_raw = 2*d + d^2 + 3;
N = size(X0, 2);

% Handle scalar vs vector t0
if isscalar(t0)
    t0_vec = t0 * ones(1, N);
else
    t0_vec = t0(:)';
end

BVP_RESID_NOISY = 1e-2;
BVP_RESID_MAX   = 1.0;
alpha1 = 0.5; alpha2 = 5.0; alpha3 = 0.5;

N_mesh = max(401, round(501 * (t_end - min(t0_vec)) / 20));

% Handle scalar vs vector compute_riccati
if isscalar(compute_riccati) && ~islogical(compute_riccati)
    cr_vec = repmat(logical(compute_riccati), 1, N);
elseif isscalar(compute_riccati)
    cr_vec = repmat(compute_riccati, 1, N);
else
    cr_vec = logical(compute_riccati);
end

% Pre-compute warm-start guesses (serial, fast)
P_ws = zeros(d, N);
for i = 1:N
    try
        p_ws0 = psi_grad(X0(:,i));
    catch
        p_ws0 = zeros(d, 1);
    end
    if norm(p_ws0) < 1e-10
        p_ws0 = ones(d, 1) * 0.1;
    end
    P_ws(:,i) = p_ws0 / max(1, norm(p_ws0)/5);
end

% Parallel output arrays
D_out = NaN(n_raw, N);
ft_out = ones(1, N, 'int8');
hV_out = false(1, N);
cV_out = false(1, N);
hH_out = false(1, N);
cH_out = false(1, N);

parfor i = 1:N
    x0 = X0(:, i);
    p_ws0 = P_ws(:, i);
    do_ric = cr_vec(i);
    t0_i = t0_vec(i);

    % Per-point mesh
    N_mesh_i = max(401, round(501 * (t_end - t0_i) / 20));
    xmesh_i = linspace(t0_i, t_end, N_mesh_i);

    % Solve BVP
    [sol, bvp_ok, bvp_clean, bvp_resid] = ...
        solve_bvp_sat(x0, xmesh_i, p_ws0, psi_grad, BVP_RESID_NOISY, BVP_RESID_MAX);

    if ~bvp_ok
        ft_out(i) = int8(1);
        continue
    end

    % Value, gradient, dtV
    p0 = sol.y(d+1:2*d, 1);
    y_end = sol.y(1:d, end);

    p4t = sol.y(d+4,:); p5t = sol.y(d+5,:); p6t = sol.y(d+6,:);
    u1t = -(p4t + (2/45)*p5t + (1/20)*p6t);
    u2t = -((1/20)*p4t + (2/3)*p5t + (1/30)*p6t);
    u3t = -((1/10)*p4t + (1/15)*p5t + (1/2)*p6t);

    yr_sq = sol.y(1,:).^2 + sol.y(2,:).^2 + sol.y(3,:).^2;
    yw_sq = sol.y(4,:).^2 + sol.y(5,:).^2 + sol.y(6,:).^2;
    u_sq  = u1t.^2 + u2t.^2 + u3t.^2;
    running = alpha1*yr_sq + alpha2*yw_sq + beta*u_sq;
    V_val = trapz(sol.x, running) + psi_val(y_end);

    y0_full = sol.y(:, 1);
    dydt0 = satellite_pmp_rhs(t0_i, y0_full);
    f0 = dydt0(1:d);
    u0 = [-(p0(4) + (2/45)*p0(5) + (1/20)*p0(6));
          -((1/20)*p0(4) + (2/3)*p0(5) + (1/30)*p0(6));
          -((1/10)*p0(4) + (1/15)*p0(5) + (1/2)*p0(6))];
    L0 = alpha1*(x0(1)^2+x0(2)^2+x0(3)^2) + alpha2*(x0(4)^2+x0(5)^2+x0(6)^2) + beta*(u0'*u0);
    dt_V = -(L0 + p0'*f0);

    % Riccati
    [ric_ok, ric_blowup, P_sol] = solve_ric_sat(sol, t0_i, t_end, psi_hess, do_ric, d);

    % Classify
    intolerable = ~bvp_clean && (bvp_resid > BVP_RESID_MAX);
    if intolerable
        ft=1; has_V=false; clean_V=false; has_H=false; clean_H=false;
    else
        has_V=true; clean_V=bvp_clean;
        if ~do_ric
            ft=10; has_H=false; clean_H=false;
        elseif bvp_clean && ric_ok, ft=0; has_H=true; clean_H=true;
        elseif bvp_clean && ric_blowup, ft=2; has_H=false; clean_H=false;
        elseif bvp_clean, ft=3; has_H=false; clean_H=false;
        elseif ric_ok, ft=4; has_H=true; clean_H=false;
        elseif ric_blowup, ft=5; has_H=false; clean_H=false;
        else, ft=6; has_H=false; clean_H=false;
        end
    end

    % Store
    row = NaN(n_raw, 1);
    row(1:d) = x0;
    if has_V
        row(d+1) = V_val;
        row(d+2:2*d+1) = p0;
        row(2*d+2) = dt_V;
    end
    if has_H
        P0 = reshape(P_sol(end,:)', d, d);
        P0 = (P0 + P0') / 2;
        row(2*d+3:2*d+2+d^2) = P0(:);
        row(n_raw) = NaN;  % dttV
    end

    D_out(:,i) = row;
    ft_out(i) = int8(ft);
    hV_out(i) = has_V;
    cV_out(i) = clean_V;
    hH_out(i) = has_H;
    cH_out(i) = clean_H;
end

Dataset = D_out;
flags = repmat(struct('ftype',1,'has_V',false,'clean_V',false,'has_H',false,'clean_H',false), 1, N);
for i = 1:N
    flags(i).ftype   = double(ft_out(i));
    flags(i).has_V   = hV_out(i);
    flags(i).clean_V = cV_out(i);
    flags(i).has_H   = hH_out(i);
    flags(i).clean_H = cH_out(i);
end
end

% =========================================================================
function [sol, bvp_ok, bvp_clean, bvp_resid] = ...
    solve_bvp_sat(x0, xmesh, p_ws0, psi_grad, BVP_RESID_NOISY, BVP_RESID_MAX)

sol = []; bvp_ok = false; bvp_clean = false; bvp_resid = Inf;
d = 6;
opts = bvpset('FJacobian', @satellite_pmp_jac, ...
              'AbsTol', 1e-8, 'RelTol', 1e-7, 'NMax', 500000);

for iguess = 1:2
    p_guess = (3 - 2*iguess) * p_ws0;
    solinit = bvpinit(xmesh, [x0; p_guess]);
    try
        lastwarn('');
        wstate = warning('off','all');
        output_str = evalc(['sol_try = bvp5c(@satellite_pmp_rhs, ' ...
            '@(ya,yb) sat_bc_local(ya,yb,x0,psi_grad), solinit, opts);']);
        warning(wstate);
    catch
        warning(wstate);
        continue
    end
    tok = regexp(output_str, 'maximum residual is ([0-9eE.+-]+)', 'tokens');
    if isempty(tok)
        [wmsg,~] = lastwarn;
        if isempty(wmsg), bvp_resid_try=0; else, bvp_resid_try=Inf; end
    else
        bvp_resid_try = max(cellfun(@(t) str2double(t{1}), tok));
    end
    if bvp_resid_try <= BVP_RESID_MAX
        sol=sol_try; bvp_ok=true;
        bvp_clean=(bvp_resid_try<=BVP_RESID_NOISY);
        bvp_resid=bvp_resid_try; break
    end
end
end

function res = sat_bc_local(ya, yb, x0, psi_grad)
d = 6;
p_term = psi_grad([yb(1);yb(2);yb(3);yb(4);yb(5);yb(6)]);
res = [ya(1:d) - x0; yb(d+1:2*d) - p_term];
end

% =========================================================================
function [ric_ok, ric_blowup, P_sol] = solve_ric_sat(sol, t0, t_end, psi_hess, do_ric, d)
ric_ok=false; ric_blowup=false; P_sol=[];
if ~do_ric, return; end
y_end = sol.y(1:d, end);
P_terminal = psi_hess(y_end);
trajs = cell(2*d, 1);
for j = 1:2*d
    trajs{j} = griddedInterpolant(sol.x, sol.y(j,:), 'pchip');
end
opts_ode = odeset('AbsTol',1e-7,'RelTol',1e-6,'Events',@ric_event);
try
    [~,P_sol,te,~,~] = ode45(@(t,P) ric_rhs(t,P,trajs,d), [t_end,t0], P_terminal(:), opts_ode);
    if isempty(te), ric_ok=true; else, ric_blowup=true; end
catch
end
end

function dPdt = ric_rhs(t, P_vec, trajs, d)
vals = zeros(2*d,1);
for j=1:2*d, vals(j)=trajs{j}(t); end
y1=vals(1);y2=vals(2);y3=vals(3);y4=vals(4);y5=vals(5);y6=vals(6);
p1=vals(7);p2=vals(8);p3=vals(9);p4=vals(10);p5=vals(11);p6=vals(12);
Hyp=satellite_Hyp(y1,y2,y3,y4,y5,y6,p1,p2,p3,p4,p5,p6);
Hpy=satellite_Hpy(y1,y2,y3,y4,y5,y6,p1,p2,p3,p4,p5,p6);
Hyy=satellite_Hyy(y1,y2,y3,y4,y5,y6,p1,p2,p3,p4,p5,p6);
Hpp=satellite_Hpp(y1,y2,y3,y4,y5,y6,p1,p2,p3,p4,p5,p6);
P=reshape(P_vec,d,d);
dPdt=(-Hyp*P - P*Hpy - P*Hpp*P - Hyy); dPdt=dPdt(:);
end

function [value,isterminal,direction] = ric_event(~,P_vec)
value=1e6-norm(P_vec); isterminal=1; direction=-1;
end
