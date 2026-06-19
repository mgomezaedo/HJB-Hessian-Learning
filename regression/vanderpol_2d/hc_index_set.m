function NN = hc_index_set(d, s)
% HC_INDEX_SET  Hyperbolic cross multi-index set via Smolyak construction.
%
% Returns all multi-indices [k1,...,kd] with k_i >= 0 such that
% prod(k_i + 1) <= s+1  (hyperbolic cross of order s).
%
% Inputs:
%   d : state space dimension
%   s : approximation order (e.g. s=4 gives q=52 for d=2)
%
% Output:
%   NN : (q x d) matrix of multi-indices, one row per basis function.
%        Row 1 is always [0,...,0] (constant term).
%        Each entry k_i >= 0 is the polynomial degree in dimension i.
%
% Example: hc_index_set(2, 4) returns 52 multi-indices.

% Build via Smolyak: get indices with prod(k_i+1) <= 2^s + 1
N1     = smolyak_elem_isotrop(d, s);   % indices start from 1
grados = prod(N1, 2);
N1(grados > 2^s + 1, :) = [];
NN     = N1 - 1;                        % shift to 0-based degrees

% Sort for consistency: constant term first, then by total degree
[~, idx] = sortrows([sum(NN,2), NN]);
NN       = NN(idx, :);

fprintf('HC index set: d=%d, s=%d -> q=%d basis functions\n', d, s, size(NN,1));
end

% =========================================================================
function Smolyak_elem_iso = smolyak_elem_isotrop(d, mu)
% Local copy of Smolyak_Elem_Isotrop (Judd, Maliar, Maliar, Valero 2014)
% Constructs multi-indices for isotropic Smolyak rule.

Smol_rule      = [];
incr_Smol_rule = [];

for j = 0:mu
    prev_incr = incr_Smol_rule;
    if j == 0
        incr_Smol_rule = ones(1, d);
    else
        m              = size(prev_incr, 1);
        incr_Smol_rule = [];
        aux            = zeros(m, d);
        for id = 1:d
            aux_new        = aux;
            aux_new(:, id) = 1;
            augmented      = prev_incr + aux_new;
            incr_Smol_rule = cat(1, incr_Smol_rule, augmented);
        end
    end
    incr_Smol_rule = unique(incr_Smol_rule, 'rows');
    Smol_rule      = cat(1, Smol_rule, incr_Smol_rule);
end

n_comb           = size(Smol_rule, 1);
Smolyak_elem_iso = [];

for i = 1:n_comb
    incr_indices = [];
    one_comb     = Smol_rule(i, :);
    for jd = 1:d
        prev_indices = incr_indices;
        if one_comb(jd) == 1
            indices_elem_jd = 1;
        elseif one_comb(jd) == 2
            indices_elem_jd = [2; 2^(one_comb(jd)-1)+1];
        else
            indices_elem_jd = (2^(one_comb(jd)-2)+2 : 2^(one_comb(jd)-1)+1)';
        end
        a = prev_indices;
        b = indices_elem_jd;
        if isempty(b)
            z = a;
        elseif isempty(a)
            z = b;
        else
            z          = [];
            a_rows     = size(a, 1);
            b_rows     = size(b, 1);
            for kk = 1:b_rows
                z = [z; [a, ones(a_rows,1)*b(kk,:)]]; %#ok<AGROW>
            end
        end
        incr_indices = z;
    end
    Smolyak_elem_iso = cat(1, Smolyak_elem_iso, incr_indices);
end
end