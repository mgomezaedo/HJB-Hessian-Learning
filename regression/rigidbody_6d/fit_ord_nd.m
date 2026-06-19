function theta = fit_ord_nd(X_t, V_t, grad_t, hess_t, NN, q, a_vec, ord, gamma1, gamma2, lambda)
% FIT_ORD_ND  Ridge regression with derivative augmentation, arbitrary dimension.
%
% Inputs:
%   X_t    : (N x d) training points
%   V_t    : (N x 1) function values
%   grad_t : (N x d) gradient values (used if ord >= 1)
%   hess_t : (N x n_hess) upper-tri Hessian values (used if ord >= 2)
%            ordering: (1,1),(2,2),...,(d,d),(1,2),(1,3),...,(d-1,d)
%   NN     : (q x d) multi-index set
%   q      : number of basis functions
%   a_vec  : (1 x d) domain half-widths
%   ord    : 0, 1, or 2
%   gamma1 : weight for first-order block
%   gamma2 : weight for second-order block
%   lambda : ridge parameter (0 for LS)

[N, d] = size(X_t);
sqN = sqrt(N);

% Evaluate basis
if ord == 0
    Phi = hc_basis_val_only(X_t, NN, a_vec);
    A0 = Phi / sqN;
    b0 = V_t / sqN;
    As = A0;
    bs = b0;
elseif ord == 1
    [Phi, dPhi] = hc_basis_grad_only(X_t, NN, a_vec);
    A0 = Phi / sqN;
    b0 = V_t / sqN;
    % Stack gradient blocks
    As = A0;
    bs = b0;
    for m = 1:d
        Am = reshape(dPhi(:, m, :), N, q) / sqN;
        bm = grad_t(:, m) / sqN;
        As = [As; gamma1*Am];
        bs = [bs; gamma1*bm];
    end
else  % ord == 2
    [Phi, dPhi, ddPhi] = hc_basis(X_t, NN, a_vec);
    n_hess = d*(d+1)/2;
    A0 = Phi / sqN;
    b0 = V_t / sqN;
    As = A0;
    bs = b0;
    % Gradient blocks
    for m = 1:d
        Am = reshape(dPhi(:, m, :), N, q) / sqN;
        bm = grad_t(:, m) / sqN;
        As = [As; gamma1*Am];
        bs = [bs; gamma1*bm];
    end
    % Hessian blocks (may be partial: hess_t can have fewer rows)
    N_hess = size(hess_t, 1);
    sqNh = sqrt(N_hess);
    for h = 1:n_hess
        Ah = reshape(ddPhi(1:N_hess, h, :), N_hess, q) / sqNh;
        bh = hess_t(:, h) / sqNh;
        As = [As; gamma2*Ah];
        bs = [bs; gamma2*bh];
    end
end

% Ridge solve
if lambda > 0
    theta = [As; sqrt(lambda)*eye(q)] \ [bs; zeros(q, 1)];
else
    theta = As \ bs;
end
end

% =====================================================================
% Helper: evaluate only values (no derivatives needed for ord=0)
% =====================================================================
function Phi = hc_basis_val_only(X, NN, a_vec)
    [N, d] = size(X);
    q = size(NN, 1);
    scale = prod(1./sqrt(a_vec));
    Xs = X ./ a_vec;
    max_deg = max(NN(:));
    leg_val = zeros(N, d, max_deg+1);
    for i = 1:d
        for m = 0:max_deg
            leg_val(:, i, m+1) = leg_legen_nor(Xs(:,i), m);
        end
    end
    Phi = zeros(N, q);
    for k = 1:q
        term = ones(N, 1);
        for i = 1:d
            term = term .* leg_val(:, i, NN(k,i)+1);
        end
        Phi(:, k) = scale * term;
    end
end

% =====================================================================
% Helper: evaluate values + gradient (no Hessian needed for ord=1)
% =====================================================================
function [Phi, dPhi] = hc_basis_grad_only(X, NN, a_vec)
    [N, d] = size(X);
    q = size(NN, 1);
    scale = prod(1./sqrt(a_vec));
    Xs = X ./ a_vec;
    max_deg = max(NN(:));
    leg_val = zeros(N, d, max_deg+1);
    leg_der = zeros(N, d, max_deg+1);
    for i = 1:d
        for m = 0:max_deg
            leg_val(:, i, m+1) = leg_legen_nor(Xs(:,i), m);
            leg_der(:, i, m+1) = (1/a_vec(i)) * legp_legen_nor(Xs(:,i), m);
        end
    end
    Phi  = zeros(N, q);
    dPhi = zeros(N, d, q);
    for k = 1:q
        powers = NN(k, :);
        term = ones(N, 1);
        for i = 1:d
            term = term .* leg_val(:, i, powers(i)+1);
        end
        Phi(:, k) = scale * term;
        for j = 1:d
            grad_term = ones(N, 1);
            for i = 1:d
                if i == j
                    grad_term = grad_term .* leg_der(:, i, powers(i)+1);
                else
                    grad_term = grad_term .* leg_val(:, i, powers(i)+1);
                end
            end
            dPhi(:, j, k) = scale * grad_term;
        end
    end
end
