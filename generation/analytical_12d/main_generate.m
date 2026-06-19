%MAIN_GENERATE  Entry point: build all datasets for the 12-D benchmark.
%   Run this file (no arguments). It writes the validation set and the
%   training set(s) to ./data using the standard schema {X,V,G,H,meta}.
%   This folder is SELF-CONTAINED: it has no external dependencies.
%
%   Usage (in MATLAB, from this folder):
%       main_generate
%
%   Pipeline:
%       main_generate -> gen_dataset_12d -> problem_12d -> {fun1dim12*, hess_to_vec}

clear; close all;

here = fileparts(mfilename('fullpath'));
addpath(here);                                  % make local functions visible
outdir = fullfile(here, 'data');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% ============================ CONFIG ============================
N_val      = 100;     % number of validation points
seed_val   = 999;     % FIXED: keep constant across every run

N_train    = 500;     % size of the example training set
seed_train = 1;

% Optional: pre-bake the full Monte-Carlo grid used by the convergence
% study. Leave false to keep generation light (regression can instead
% regenerate training sets on the fly via gen_dataset_12d). Set true for
% a fully pre-computed, self-contained dataset (writes to data/mc/).
GENERATE_MC_GRID = false;
N_grid = [50 100 150 200 300 400 500 700 1000];
n_runs = 100;         % MC repetitions per N
% ================================================================

% --- validation set ---
val = gen_dataset_12d(N_val, seed_val);                          %#ok<NASGU>
save(fullfile(outdir, 'val_12d.mat'), '-struct', 'val');
fprintf('val_12d.mat    N=%d\n', N_val);

% --- example training set ---
train = gen_dataset_12d(N_train, seed_train);                    %#ok<NASGU>
save(fullfile(outdir, 'train_12d.mat'), '-struct', 'train');
fprintf('train_12d.mat  N=%d\n', N_train);

% --- optional Monte-Carlo grid ---
if GENERATE_MC_GRID
    mcdir = fullfile(outdir, 'mc');
    if ~exist(mcdir, 'dir'); mkdir(mcdir); end
    for N = N_grid
        for run = 1:n_runs
            D = gen_dataset_12d(N, run);                         %#ok<NASGU>
            fn = sprintf('train_N%04d_run%03d.mat', N, run);
            save(fullfile(mcdir, fn), '-struct', 'D');
        end
        fprintf('MC grid N=%d done (%d runs)\n', N, n_runs);
    end
end

fprintf('Done. Output in %s\n', outdir);
