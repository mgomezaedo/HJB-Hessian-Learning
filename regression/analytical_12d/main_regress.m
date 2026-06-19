%MAIN_REGRESS  Entry point: fit the value function, report validation error.
%   Loads the datasets produced by the generation stage, builds the chosen
%   polynomial basis, fits the coefficients with fit_ord_nd, and computes
%   the relative L2 / H1 / H2 validation errors.
%
%   Usage (in MATLAB, from this folder):
%       main_regress

clear; close all;
here = fileparts(mfilename('fullpath')); addpath(here);

% ============================ CONFIG ============================
% Folder with the .mat datasets written by generation/analytical_12d:
data_dir = fullfile(here, '..', '..', 'generation', 'analytical_12d', 'data');
train_file = 'train_12d.mat';
val_file   = 'val_12d.mat';

hc_level = 4;        % hyperbolic-cross level  ->  basis HC(hc_level)
ord      = 2;        % 0 = value only, 1 = + gradient, 2 = + Hessian
lambda   = 1e-8;     % ridge parameter (0 = plain least squares)
gamma1   = 1.0;      % weight of the gradient block
gamma2   = 1.0;      % weight of the Hessian block
% ================================================================

% --- load data ---
Dtr  = load(fullfile(data_dir, train_file));
Dval = load(fullfile(data_dir, val_file));
d     = Dtr.meta.d;
a_vec = (Dtr.meta.ub - Dtr.meta.lb) / 2;          % domain half-widths

% --- build basis ---
NN = hc_index_set(d, hc_level);
q  = size(NN, 1);
fprintf('Basis HC(%d): q = %d functions, d = %d, N_train = %d\n', ...
        hc_level, q, d, size(Dtr.X,1));

% --- derivative inputs according to the chosen order ---
switch ord
    case 0, G = [];      H = [];
    case 1, G = Dtr.G;   H = [];
    case 2, G = Dtr.G;   H = Dtr.H;
end

% --- fit coefficients ---
theta = fit_ord_nd(Dtr.X, Dtr.V, G, H, NN, q, a_vec, ord, gamma1, gamma2, lambda);
fprintf('Fitted theta: %d coefficients, ||theta|| = %.4e\n', numel(theta), norm(theta));

% --- validation error ---
[eL2, eH1, eH2] = val_error(theta, NN, a_vec, Dval);
fprintf('\nValidation (relative) errors, order %d:\n', ord);
fprintf('  L2 = %.4e\n', eL2);
fprintf('  H1 = %.4e\n', eH1);
fprintf('  H2 = %.4e\n', eH2);

% theta + NN are the result. Uncomment to save:
% save(fullfile(here,'coeffs_12d.mat'), 'theta', 'NN', 'a_vec', 'ord', 'hc_level');
