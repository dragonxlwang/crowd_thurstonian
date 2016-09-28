function [s, eta, f] = pair_bt(data, lambda, qnum, wnum, dnum, ...
                          dump_file_path, max_iter)
%PAIR_BT Summary of this function goes here
%   Detailed explanation goes here
    data_size = size(data);
    inst_num = data_size(1);
    data_by_w = cell(1, wnum);
    data_by_q = cell(1, qnum);
    iter = 0;
    for t = 1:inst_num
        k = data(t, 1);
        l = data(t, 2);
        data_by_w{k} = [data_by_w{k}; data(t, :)];
        data_by_q{l} = [data_by_q{l}; data(t, :)];
    end
    eta = ones(length(data_by_w), 1);
    f_old = -inf;
    while(true)
        iter = iter + 1;
        if(iter > max_iter)
            break;
        end
        fprintf('iteration [%d]\n', iter);
        s = opt_s(data_by_q, lambda, eta, dnum);
        eta = opt_eta(data_by_w, lambda, s);
%         if(mod(iter, 5) == 0)
            f = fobj(data, lambda, s, eta);
            fprintf('objective [%f]\n', f);
%             fprintf('writing eta file: %s\n', fp);
            fp = sprintf('%s_%s_final', dump_file_path, 's');
            csvwrite(fp, cell_to_matrix(s, dnum));
%             fprintf('writing s file: %s\n', fp);
            fp = sprintf('%s_%s_final', dump_file_path, 'eta');
            csvwrite(fp, eta);
%         end
        if(abs(f - f_old) / abs(f) < 1e-5)
            break;
        end
        f_old = f;
    end
end

function m = cell_to_matrix(s, dnum)
    m = zeros(sum(dnum), 3);
    t = 1;
    for l = 1:length(s)
        for i = 1:length(s{l})
            m(t, :) = [l, i, s{l}(i)];
            t = t + 1;
        end
    end            
end

function s = opt_s(data_by_q, lambda, eta, dnum)
    s = cell(1, length(data_by_q));
    for l = 1:length(s)
        data_l = data_by_q{l};
        options.display = 'none';
%         fprintf('optimizing s_%d', l);
        s_l = minFunc(@(s_l) func_by_q(data_l, lambda, s_l, eta), ...
                zeros(dnum(l), 1), options);
        s{l} = s_l;
    end
end

function eta = opt_eta(data_by_w, lambda, s)
    eta = zeros(length(data_by_w), 1);
    for k = 1:length(eta)
        data_k = data_by_w{k};
        options.display = 'none';
%         fprintf('optimizing eta_%d', k);
        eta_k = minFunc(@(eta_k) func_by_w(data_k, lambda, s, eta_k), ...
                1.0, options);
        if(eta_k > 1)
            eta(k) = 1;
        elseif(eta_k < 0)
            eta(k) = 0;
        else
            eta(k) = eta_k;        
        end
%         fprintf('optimizing eta_%d = %.4f => %.4f\n', k, eta_k, eta(k));
    end
end

function [f, g] = func_by_q(data_l, lambda, s_l, eta)
    g = zeros(length(s_l), 1);
    f = 0.0;
    data_size = size(data_l);
    inst_num = data_size(1);
    for t = 1:inst_num
        k = data_l(t, 1);
        i = data_l(t, 3);
        j = data_l(t, 4);
        gg = gobj_si_per_row(s_l(i), s_l(j), eta(k));
        g(i) = g(i) + gg;
        g(j) = g(j) - gg;
        f = f + fobj_per_row(s_l(i), s_l(j), eta(k));
    end
    g = g + lambda * ( ones(length(s_l), 1) - 2 * sigm(s_l) );
    f = -f;
    g = -g;
end

function [f,g] = func_by_w(data_k, lambda, s, eta_k)
    g = 0;
    f = 0.0;
    data_size = size(data_k);
    inst_num = data_size(1);
    for t = 1:inst_num
        l = data_k(t, 2);
        i = data_k(t, 3);
        j = data_k(t, 4);
        g = g + gobj_eta_k_per_row(s{l}(i), s{l}(j), eta_k);
        f = f + fobj_per_row(s{l}(i), s{l}(j), eta_k);
%         fprintf('%d, %d, %f\n', l, i, s{l}(i));
%         fprintf('%d, %d, %f\n', l, j, s{l}(j));
%         fprintf('eta_k = %f\n', eta_k);
%         fprintf('f = %f, df = %f\n', f, fobj_per_row(s{l}(i), s{l}(j), eta_k));
    end    
    f = -f;
    g = -g;
end

function f = fobj_per_row(si, sj, eta_k)
    f = log((2*eta_k - 1) * sigm(si - sj) + 1 - eta_k);
end

function f = fobj(data, lambda, s, eta)
    data_size = size(data);
    inst_num = data_size(1);
    f = 0.0;
    for t = 1:inst_num
        k = data(t,1);
        l = data(t,2);
        i = data(t,3);
        j = data(t,4);
        si = s{l}(i);
        sj = s{l}(j);
        f = f + fobj_per_row(si, sj, eta(k));
    end
    for l = 1:length(s)
        for i=1:length(s{l})
            f = f + lambda * (log(sigm(s{l}(i))) + log(sigm(-s{l}(i))));
        end
    end
end

function g = gobj_eta_k_per_row(si, sj, eta_k)
    s = sigm(si - sj);
    g = (2 * s - 1) / ((2 * eta_k - 1) * s + 1 - eta_k);
end

function g = gobj_si_per_row(si, sj, eta_k)
    s = sigm(si - sj);
    a = (2 * eta_k - 1) * s;
    g = a * (1 - s) / (a + 1 - eta_k);
end

function g = gobj_sj_per_row(si, sj, eta_k)
    g = -gobj_si_per_row(si, sj, eta_k);
end

function f = sigm(x)
    f = 1.0 ./ (1 + exp(-x));
end