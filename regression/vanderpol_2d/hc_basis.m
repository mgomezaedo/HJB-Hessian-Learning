function [Phi, dPhi, ddPhi] = hc_basis(X, NN, a_vec)
% HC_BASIS  Evaluate hyperbolic cross polynomial basis and derivatives.
%
% Evaluates the normalized Legendre basis on [-a_i, a_i] at N points.
% Normalization: phi_k(x) = scale * prod_i L_{k_i}(x_i/a_i)
% where scale = prod(1/sqrt(a_i)) ensures L2-orthonormality on the domain.
%
% Inputs:
%   X     : (N x d) matrix of evaluation points
%   NN    : (q x d) matrix of multi-indices from hc_index_set
%   a_vec : (1 x d) domain half-widths (domain is [-a_i, a_i] per dim)
%
% Outputs:
%   Phi   : (N x q)         basis values
%   dPhi  : (N x d x q)     spatial gradients  d(phi_k)/d(x_i)
%   ddPhi : (N x d*(d+1)/2 x q)  upper-triangular Hessian entries
%             ordering: (1,1),(2,2),...,(d,d),(1,2),(1,3),...,(d-1,d)
%             i.e. diagonal first then upper off-diagonal pairs
%
% All outputs are (N x ...) so rows correspond to sample points.

[N, d] = size(X);
q      = size(NN, 1);
scale  = prod(1./sqrt(a_vec));   % L2-orthonormality correction for domain
n_hess = d*(d+1)/2;

% Normalize x to [-1,1] in each dimension
Xs = X ./ a_vec;                 % (N x d), broadcasts over rows

% Precompute univariate basis values and derivatives at all points
% leg_val{i}(n,m)  = L_m(Xs(n,i))
% leg_der{i}(n,m)  = L'_m(Xs(n,i))  [wrt x_i, includes 1/a_i factor]
% leg_der2{i}(n,m) = L''_m(Xs(n,i)) [wrt x_i, includes 1/a_i^2 factor]
max_deg = max(NN(:));
leg_val  = zeros(N, d, max_deg+1);
leg_der  = zeros(N, d, max_deg+1);
leg_der2 = zeros(N, d, max_deg+1);

for i = 1:d
    for m = 0:max_deg
        leg_val(:, i, m+1)  = leg_legen_nor(Xs(:,i), m);
        leg_der(:, i, m+1)  = (1/a_vec(i))  * legp_legen_nor(Xs(:,i), m);
        leg_der2(:, i, m+1) = (1/a_vec(i)^2)* legpp_legen_nor(Xs(:,i), m);
    end
end

% Allocate outputs
Phi   = zeros(N, q);
dPhi  = zeros(N, d, q);
ddPhi = zeros(N, n_hess, q);

for k = 1:q
    powers = NN(k, :);   % (1 x d) degree vector for this basis function

    % --- Value: phi_k(x) = scale * prod_i L_{ki}(xi/ai) ---
    term = ones(N, 1);
    for i = 1:d
        term = term .* leg_val(:, i, powers(i)+1);
    end
    Phi(:, k) = scale * term;

    % --- Gradient: d(phi_k)/d(xj) = scale * L'_kj(xj/aj)/aj * prod_{i~=j} L_ki(xi/ai) ---
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

    % --- Hessian: diagonal and upper off-diagonal ---
    idx_h = 1;
    % Diagonal: d^2(phi_k)/d(xi)^2
    for i = 1:d
        hess_term = ones(N, 1);
        for ii = 1:d
            if ii == i
                hess_term = hess_term .* leg_der2(:, ii, powers(ii)+1);
            else
                hess_term = hess_term .* leg_val(:, ii, powers(ii)+1);
            end
        end
        ddPhi(:, idx_h, k) = scale * hess_term;
        idx_h = idx_h + 1;
    end
    % Off-diagonal: d^2(phi_k)/d(xi)d(xj), i < j
    for i = 1:d
        for j = i+1:d
            hess_term = ones(N, 1);
            for ii = 1:d
                if ii == i || ii == j
                    hess_term = hess_term .* leg_der(:, ii, powers(ii)+1);
                else
                    hess_term = hess_term .* leg_val(:, ii, powers(ii)+1);
                end
            end
            ddPhi(:, idx_h, k) = scale * hess_term;
            idx_h = idx_h + 1;
        end
    end
end
end