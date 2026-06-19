function NN = hc_enriched_index_set(d, s, max_total_deg, max_q)
% HC_ENRICHED_INDEX_SET  Hyperbolic cross + total-degree enrichment.
%
% Starts from the HC(s) index set and adds multi-indices from the
% total-degree set {k : sum(k) <= max_total_deg} that are NOT already
% in the HC set.  The result is a basis richer than HC(s) but without
% the extreme single-variable degrees that HC(s+1) introduces.
%
% For d=2, s=4:
%   HC(4) alone:  q=52,  max univariate degree = 16
%   HC(4) + TD(6): q~72,  max univariate degree = 16 (no increase!)
%   HC(4) + TD(7): q~84,  max univariate degree = 16
%   HC(4) + TD(8): q~98,  max univariate degree = 16
%   HC(5) alone:  q=123, max univariate degree = 32 (bad!)
%
% The enrichment adds cross-terms like (3,3), (4,2), (2,4), (3,4), etc.
% that the HC misses but are needed for capturing V(x) in corners.
%
% Optional max_q: if the enriched set exceeds max_q, truncate by keeping
% the indices with smallest total degree (greedy, preserves HC base).
%
% Inputs:
%   d             : state space dimension
%   s             : HC approximation order (base set)
%   max_total_deg : maximum total degree for enrichment (e.g. 6, 7, 8)
%   max_q         : (optional) maximum number of basis functions
%
% Output:
%   NN : (q x d) matrix of multi-indices, sorted by total degree

if nargin < 4, max_q = Inf; end

% --- Build HC base ---
NN_hc = hc_index_set_silent(d, s);

% --- Build total-degree set up to max_total_deg ---
% All k in N_0^d with sum(k) <= max_total_deg
NN_td = build_total_degree(d, max_total_deg);

% --- Merge: union of HC and TD, remove duplicates ---
NN_all = unique([NN_hc; NN_td], 'rows');

% --- Sort by total degree, then lexicographic ---
td = sum(NN_all, 2);
[~, idx] = sortrows([td, NN_all]);
NN_all = NN_all(idx, :);

% --- Truncate if needed ---
if size(NN_all, 1) > max_q
    NN_all = NN_all(1:max_q, :);
end

NN = NN_all;

% --- Report ---
max_deg_per_dim = max(NN, [], 1);
fprintf('HC-enriched index set: d=%d, s=%d, max_td=%d -> q=%d (HC base=%d)\n', ...
    d, s, max_total_deg, size(NN,1), size(NN_hc,1));
fprintf('  Max degree per dim: [%s]  Max total degree: %d\n', ...
    num2str(max_deg_per_dim), max(sum(NN,2)));
end

% =========================================================================
function NN_td = build_total_degree(d, s_td)
% Build all multi-indices k in N_0^d with sum(k) <= s_td.
% For d=2 this is (s_td+1)*(s_td+2)/2 indices.
if d == 1
    NN_td = (0:s_td)';
    return;
end

% Recursive construction for general d
NN_td = [];
for k = 0:s_td
    % First component = k, rest has sum <= s_td - k
    rest = build_total_degree(d-1, s_td - k);
    NN_td = [NN_td; k*ones(size(rest,1),1), rest]; %#ok<AGROW>
end
end

% =========================================================================
function NN = hc_index_set_silent(d, s)
% Same as hc_index_set but without fprintf (for use inside this function)
N1     = smolyak_elem_isotrop(d, s);
grados = prod(N1, 2);
N1(grados > 2^s + 1, :) = [];
NN     = N1 - 1;
[~, idx] = sortrows([sum(NN,2), NN]);
NN       = NN(idx, :);
end

% =========================================================================
function Smolyak_elem_iso = smolyak_elem_isotrop(d, mu)
Smol_rule = []; incr_Smol_rule = [];
for j = 0:mu
    prev_incr = incr_Smol_rule;
    if j == 0
        incr_Smol_rule = ones(1, d);
    else
        m = size(prev_incr, 1);
        incr_Smol_rule = [];
        aux = zeros(m, d);
        for id = 1:d
            aux_new = aux; aux_new(:, id) = 1;
            incr_Smol_rule = cat(1, incr_Smol_rule, prev_incr + aux_new);
        end
    end
    incr_Smol_rule = unique(incr_Smol_rule, 'rows');
    Smol_rule = cat(1, Smol_rule, incr_Smol_rule);
end
n_comb = size(Smol_rule, 1);
Smolyak_elem_iso = [];
for i = 1:n_comb
    incr_indices = [];
    one_comb = Smol_rule(i, :);
    for jd = 1:d
        prev_indices = incr_indices;
        if one_comb(jd)==1, indices_elem_jd = 1;
        elseif one_comb(jd)==2, indices_elem_jd = [2; 2^(one_comb(jd)-1)+1];
        else, indices_elem_jd = (2^(one_comb(jd)-2)+2 : 2^(one_comb(jd)-1)+1)';
        end
        a = prev_indices; b = indices_elem_jd;
        if isempty(b), z=a; elseif isempty(a), z=b;
        else
            z = [];
            for kk = 1:size(b,1)
                z = [z; [a, ones(size(a,1),1)*b(kk,:)]]; %#ok<AGROW>
            end
        end
        incr_indices = z;
    end
    Smolyak_elem_iso = cat(1, Smolyak_elem_iso, incr_indices);
end
end
