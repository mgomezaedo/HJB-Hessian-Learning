%MAIN_GENERATE  Entry point: build the 6-D satellite datasets (train + val).
%   Run this file (no arguments). It writes a training set and a validation
%   set to ./data using the standard schema {X, V, G, H, meta}, both built
%   with the SAME parallel PMP + Riccati engine.
%
%   Usage (in MATLAB, from this folder):
%       main_generate
%
%   Pipeline:
%       main_generate -> gen_dataset_rb6d -> solve_pmp_riccati_par
%                        -> {problem_satellite6d, satellite_*}
%
%   WARNING: 6-D boundary-value solves over the long horizon (T=20) are
%   expensive. Generation is parallelised (parfor over points); a parallel
%   pool speeds it up a lot. Expect some points to be discarded (M < N).

clear; close all;
here = fileparts(mfilename('fullpath')); addpath(here);
outdir = fullfile(here, 'data');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% ============================ CONFIG ============================
N_train    = 500;    % requested training points (kept count will be lower)
seed_train = 1;

N_val      = 500;    % requested validation points
seed_val   = 999;    % fixed

USE_PARALLEL_POOL = true;   % start a local pool if none is open
% ================================================================

if USE_PARALLEL_POOL && isempty(gcp('nocreate'))
    try
        parpool('local');
    catch
        warning('Could not start a parallel pool; parfor will run serially.');
    end
end

% --- validation set ---
val = gen_dataset_rb6d(N_val, seed_val);           %#ok<NASGU>
save(fullfile(outdir, 'val_rb6d.mat'), '-struct', 'val');
fprintf('Wrote %s\n', fullfile(outdir, 'val_rb6d.mat'));

% --- training set ---
train = gen_dataset_rb6d(N_train, seed_train);     %#ok<NASGU>
save(fullfile(outdir, 'train_rb6d.mat'), '-struct', 'train');
fprintf('Wrote %s\n', fullfile(outdir, 'train_rb6d.mat'));

fprintf('Done. Output in %s\n', outdir);
