function [eL2, eH1, eH2] = val_error(theta, NN, a_vec, Dval)
%VAL_ERROR  Relative L2/H1/H2 errors of a fitted model on a validation set.
%   [eL2, eH1, eH2] = VAL_ERROR(theta, NN, a_vec, Dval)
%
%   Inputs:
%     theta : (q x 1) fitted coefficients
%     NN    : (q x d) multi-index set (same used for the fit)
%     a_vec : (1 x d) domain half-widths
%     Dval  : struct with fields X, V, G, H (a validation dataset)
%
%   Outputs (relative errors):
%     eL2 : value error      ||Vhat - V|| / ||V||
%     eH1 : gradient error   ||ghat - G|| / ||G||
%     eH2 : Hessian error    ||hhat - H|| / ||H||

    [Nval, d] = size(Dval.X);
    q = size(NN, 1);
    n_hess = d*(d+1)/2;

    [Phi, dPhi, ddPhi] = hc_basis(Dval.X, NN, a_vec);

    % --- predicted value / gradient / Hessian ---
    Vhat = Phi * theta;

    ghat = zeros(Nval, d);
    for m = 1:d
        ghat(:, m) = reshape(dPhi(:, m, :), Nval, q) * theta;
    end

    hhat = zeros(Nval, n_hess);
    for h = 1:n_hess
        hhat(:, h) = reshape(ddPhi(:, h, :), Nval, q) * theta;
    end

    % --- reference norms (from the validation labels) ---
    norm_V = sqrt(mean(Dval.V.^2));
    norm_g = sqrt(mean(sum(Dval.G.^2, 2)));
    norm_h = sqrt(mean(sum(Dval.H.^2, 2)));

    % --- relative errors ---
    eL2 = sqrt(mean((Vhat - Dval.V).^2))          / norm_V;
    eH1 = sqrt(mean(sum((ghat - Dval.G).^2, 2)))  / norm_g;
    eH2 = sqrt(mean(sum((hhat - Dval.H).^2, 2)))  / norm_h;
end
