%MAIN_REGRESS  Entry point: fit the VdP value function, report validation error.
%   Loads the datasets produced by generation/vanderpol_2d, builds the chosen
%   polynomial basis, fits the coefficients with fit_ord_nd, and computes the
%   relative L2 / H1 / H2 validation errors. Self-contained.
%
%   Usage (in MATLAB, from this folder):
%       main_regress

clear; close all;
here = fileparts(mfilename('fullpath')); addpath(here);

% ============================ CONFIG ============================
% Folder with the .mat datasets written by generation/vanderpol_2d:
data_dir = fullfile(here, '..', '..', 'generation', 'vanderpol_2d', 'data');
train_file = 'train_vdp2d.mat';
val_file   = 'val_vdp2d.mat';

% --- basis choice ---
basis_type = 'HC';   % 'HC' = hyperbolic cross, 'enriched' = HC + total-degree
hc_level   = 4;      % HC level
td_level   = 12;     % total-degree level (only used if basis_type='enriched')

% --- method / regularisation ---
ord    = 2;          % 0 = value only, 1 = + gradient, 2 = + Hessian
lambda = 1e-10;      % ridge parameter
gamma1 = 1.0;        % weight of the gradient block
gamma2 = 1.0;        % weight of the Hessian block
margin = 0.15;       % basis domain extended by this fraction beyond the box
% ================================================================

% --- load data ---
Dtr  = load(fullfile(data_dir, train_file));
Dval = load(fullfile(data_dir, val_file));
d      = Dtr.meta.d;
a_half = (Dtr.meta.ub - Dtr.meta.lb) / 2;     % box half-widths
a_vec  = a_half * (1 + margin);               % extended basis half-widths

% --- build basis ---
switch basis_type
    case 'HC'
        NN = hc_index_set(d, hc_level);
        bname = sprintf('HC(%d)', hc_level);
    case 'enriched'
        NN = hc_enriched_index_set(d, hc_level, td_level);
        bname = sprintf('HC(%d)+TD(%d)', hc_level, td_level);
    otherwise
        error('basis_type must be ''HC'' or ''enriched''');
end
q = size(NN, 1);
fprintf('Basis %s: q = %d functions, d = %d, N_train = %d\n', ...
        bname, q, d, size(Dtr.X,1));

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
% save(fullfile(here,'coeffs_vdp2d.mat'), 'theta', 'NN', 'a_vec', 'ord', 'basis_type');
