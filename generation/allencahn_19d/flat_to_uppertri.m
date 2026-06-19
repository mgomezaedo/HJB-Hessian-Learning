function htri = flat_to_uppertri(Hflat, d)
% Convert flat (d^2 × N) Hessians (col-major) to upper-tri (N × d*(d+1)/2).
% Order: diag first (1,1)..(d,d), then upper off-diag in row-major.
n_samples = size(Hflat, 2);
n_hess = d*(d+1)/2;
htri = zeros(n_samples, n_hess);
for j = 1:n_samples
    H = reshape(Hflat(:, j), d, d);
    idx = 1;
    for i = 1:d
        htri(j, idx) = H(i, i);
        idx = idx + 1;
    end
    for i = 1:d
        for k = i+1:d
            htri(j, idx) = H(i, k);
            idx = idx + 1;
        end
    end
end
end