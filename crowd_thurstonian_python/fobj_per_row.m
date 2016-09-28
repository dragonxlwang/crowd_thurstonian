function f = fobj_per_row(si, sj, eta_k)
    f = log((2*eta_k - 1) * sigm(si - sj) + 1 - eta_k);
end