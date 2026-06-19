function value = computeValueFunction(sol)
    % Compute V*(x) = J*(x) along the optimal trajectory using:
    %   J* = ∫_0^T [Δx · ||y||² + (1/4β) · ||B'p||²] dt
    %
    % BUG FIX (Phase 3): previous version computed β·||p||² instead of
    % (1/4β)·||B'p||², which is the running cost under optimal control
    % u* = -(1/2β) B'p.
    
    beta   = 0.01;
    N_grid = 20;
    d      = N_grid - 1;
    Deltax = 2 / N_grid;
    
    % B_M matrix (must match bvpfcn.m and Hpp/Hyy/etc.)
    omega_1 = zeros(d, 1); omega_1([4, 5])      = 1;
    omega_2 = zeros(d, 1); omega_2([9, 10, 11]) = 1;
    omega_3 = zeros(d, 1); omega_3([15, 16])    = 1;
    B_M = [omega_1, omega_2, omega_3];
    
    % Trajectories: y(t) in rows 1:d, p(t) in rows d+1:2d
    y_traj = sol.y(1:d, :);            % d × n_t
    p_traj = sol.y(d+1:2*d, :);        % d × n_t
    
    % Running cost integrand
    state_cost   = Deltax * sum(y_traj.^2, 1);        % 1 × n_t
    BtP          = B_M' * p_traj;                      % 3 × n_t
    control_cost = (1/(4*beta)) * sum(BtP.^2, 1);     % 1 × n_t
    
    value = trapz(sol.x, state_cost + control_cost);
end