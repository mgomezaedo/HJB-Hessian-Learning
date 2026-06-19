%MAIN_GENERATE  Entry point: build the Van der Pol datasets (train + val).
%   Run this file (no arguments). It writes a training set and a validation
%   set to ./data using the standard schema {X, V, G, H, meta}, both
%   generated with the SAME PMP + Riccati engine.
%
%   Usage (in MATLAB, from this folder):
%       main_generate
%
%   Pipeline:
%       main_generate -> gen_dataset_vdp2d -> {problem_vdp2d, solve_pmp_riccati}
%
%   NOTE: each point is a boundary-value solve, so this is much slower than
%   the analytical benchmark. A few hundred points takes a few minutes.

clear; close all;
here = fileparts(mfilename('fullpath')); addpath(here);
outdir = fullfile(here, 'data');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% ============================ CONFIG ============================
N_train    = 500;    % requested training points (kept count will be lower)
seed_train = 1;

N_val      = 500;    % requested validation points
seed_val   = 999;    % fixed
% ================================================================

% --- validation set ---
val = gen_dataset_vdp2d(N_val, seed_val);          %#ok<NASGU>
save(fullfile(outdir, 'val_vdp2d.mat'), '-struct', 'val');
fprintf('Wrote %s\n', fullfile(outdir, 'val_vdp2d.mat'));

% --- training set ---
train = gen_dataset_vdp2d(N_train, seed_train);    %#ok<NASGU>
save(fullfile(outdir, 'train_vdp2d.mat'), '-struct', 'train');
fprintf('Wrote %s\n', fullfile(outdir, 'train_vdp2d.mat'));

fprintf('Done. Output in %s\n', outdir);
