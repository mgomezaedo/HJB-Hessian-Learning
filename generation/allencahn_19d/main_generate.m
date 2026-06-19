%MAIN_GENERATE  Entry point: build the 19-D Allen-Cahn datasets (train + val).
%   Run this file (no arguments). It writes a training set and a validation
%   set to ./data using the standard schema {X, V, G, H, meta}, both built
%   with the SAME parallel Pontryagin + Riccati engine.
%
%   Usage (in MATLAB, from this folder):
%       main_generate
%
%   Pipeline:
%       main_generate -> gen_dataset_ac19d
%                        -> {solvePontryagin (bvpfcn/bcfcn/guess),
%                            computeValueFunction,
%                            solveRiccati (rhsric, H**_VdP),
%                            flat_to_uppertri, problem_ac19d}
%
%   WARNING: 19-D boundary-value solves over the horizon are expensive even
%   in parallel. A few hundred points takes several minutes; expect some
%   points to be discarded (M < N).

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
        parpool('Processes');
    catch
        warning('Could not start a parallel pool; parfor will run serially.');
    end
end

% --- validation set ---
val = gen_dataset_ac19d(N_val, seed_val);          %#ok<NASGU>
save(fullfile(outdir, 'val_ac19d.mat'), '-struct', 'val', '-v7.3');
fprintf('Wrote %s\n', fullfile(outdir, 'val_ac19d.mat'));

% --- training set ---
train = gen_dataset_ac19d(N_train, seed_train);    %#ok<NASGU>
save(fullfile(outdir, 'train_ac19d.mat'), '-struct', 'train', '-v7.3');
fprintf('Wrote %s\n', fullfile(outdir, 'train_ac19d.mat'));

fprintf('Done. Output in %s\n', outdir);
