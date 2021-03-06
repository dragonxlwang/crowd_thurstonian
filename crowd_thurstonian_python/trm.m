function trm(data, qnum, wnum, dnum, ...
                    dump_file_path, max_iter)
    burnnum = 100;
    sampnum = 100;
    [ps, s, delta] = initialize(data, qnum, wnum, dnum);
    s_mat = csvread(['/home/xwang95/exp/tpp/mq2008agg_small/' ...
                     'mq2008agg_small_trm_s_bc_init.mat.txt']);
%     for t = 1:length(s_mat) % initialize s
%         l = int8(s_mat(t,1));
%         i = int8(s_mat(t,2));
%         s{l}(i) = s_mat(t,3);
%     end
%     delta = ones(qnum, 1) + 1e-2; % initialize delta
%     for t = 1:length(data) % initialize ps
%         k = data(t, 1);
%         l = data(t, 2);
%         i = data(t, 3);
%         prev = data(t, 4);
%         after = data(t, 5);
%         if(prev == -1)
%             ps{l}(k,i) = s{l}(i);
%         else
%             ub = (ps{l}(k, prev) - s{l}(i)) / 0.1;
%             ps{l}(k,i) = rtnorm(-1e100, ub) * 0.1 + s{l}(i);
%         end
%     end
%     
    delta = ones(qnum, 1) + 1e-2; % initialize delta
    for t = length(data):-1:1 % initialize ps
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        prev = data(t, 4);
        after = data(t, 5);
        if(after == -1)
            ps{l}(k,i) = 0.0;
        else
            ps{l}(k,i) = 1.0 + ps{l}(k, after);
        end
    end
    sm = cell(qnum, 1); % initialize s
    s = cell(qnum, 1);
    for l = 1:qnum
        sm{l} = zeros(dnum(l), 1);
        s{l} = zeros(dnum(l), 1);
    end
    for t = length(data):-1:1 
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        s{l}(i) = s{l}(i) + ps{l}(k,i);
        sm{l}(i) = sm{l}(i) + 1.0;
    end
    for l = 1:qnum
        for i = 1:dnum(l)
            s{l}(i) = s{l}(i) / sm{l}(i);
        end
    end

    epoch = 1;
    while(true)
        if(epoch > max_iter)
            fprintf('==>finishing<==\n');
            break;
        end
        fprintf('iteration [%d]\n', epoch);
        fprintf(' |-> sampling\n');
        if(epoch == 1)
            burnnum = 100;
        else
            burnnum = 100;
        end
        psseq = posterior_sampling_phase(data, delta, s, ps, ...
                qnum, wnum, dnum, burnnum, sampnum);
        fprintf(' |-> updating\n');
        [s, delta, ps] = param_update(data, psseq, ... 
                        qnum, wnum, dnum, sampnum);
        fprintf(' |-> dumping\n');
        fp = sprintf('%s%s_%d', dump_file_path, 's', epoch+100);
        fprintf('     [file]: %s\n', fp);
        csvwrite(fp, cell_to_matrix(s, dnum));
        fp = sprintf('%s%s_%d', dump_file_path, 'delta', epoch+100);
        fprintf('     [file]: %s\n', fp);
        csvwrite(fp, delta);
        epoch = epoch + 1;
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

function psseq = posterior_sampling_phase(data, delta, s, ps, ...
                qnum, wnum, dnum, burnnum, sampnum)
     fprintf('   |-> burning iteration ');
     for iter = 1:burnnum
         if(mod(iter, int8(burnnum/10)) == 0)
             fprintf('*');
         end
         ps = posterior_sampling_iter(data, delta, ...
                qnum, wnum, dnum, s, ps);
     end
     fprintf('\n');
     psseq = cell(1, sampnum);
     fprintf('   |-> sampling iteration ');
     for iter = 1:sampnum
         if(mod(iter, int8(sampnum/10)) == 0)
             fprintf('*');
         end
         psseq{iter} = posterior_sampling_iter(data, delta, ...
                        qnum, wnum, dnum, s, ps);
     end
     fprintf('\n');
end

function [s, delta, ps] = param_update(data, psseq, ... 
                        qnum, wnum, dnum, sampnum)
    psmom1 = cell(1, qnum);
    psmom2 = cell(1, qnum);
    for l = 1:qnum
        psmom1{l} = zeros(wnum,dnum(l));
        psmom2{l} = zeros(wnum,dnum(l));
    end
    for t = 1:length(data)
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        arr = zeros(sampnum, 1);
        for iter = 1:sampnum
            arr(iter) = psseq{iter}{l}(k,i);
        end
        psmom1{l}(k,i) = sum(arr) / sampnum;
        psmom2{l}(k,i) = sum(arr .^ 2) / sampnum;
    end
    s = cell(1, qnum);
    sm = cell(1, qnum);
    for l = 1:qnum
        s{l} = zeros(dnum(l), 1);
        sm{l} = zeros(dnum(l), 1);
    end
    for t = 1:length(data)
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        s{l}(i) = s{l}(i) + psmom1{l}(k,i);
        sm{l}(i) = sm{l}(i) + 1;
    end
    for l = 1:qnum
        for i = 1:dnum(l)
            s{l}(i) = s{l}(i) / sm{l}(i);
        end
    end
    delta = zeros(qnum, 1);
    deltam = zeros(qnum, 1);
    for t = 1:length(data)
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        delta(l) = delta(l) + psmom2{l}(k,i) + s{l}(i)^2 ...
                    - 2*psmom1{l}(k,i)*s{l}(i);
        deltam(l) = deltam(l) + 1;
    end
    for l = 1:qnum
        delta(l) = sqrt(delta(l) / deltam(l));
    end
    [ps, s, delta] = normalize(psmom1, s, delta);
end

function nps = posterior_sampling_iter(data, delta, qnum, wnum, dnum, ...
                                        s, ps)
    % ps{l}(k,i)
    % s{l}(i)
    nps = cell(1, qnum);
    for l = 1:qnum
        for i = 1:dnum(l)
            for k = 1:wnum
                nps{l}(k,i) = ps{l}(k,i);
            end
        end
    end
    for t = 1:length(data)
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        prev = data(t, 4);
        after = data(t, 5);
        if(prev == -1)
            ps_prev = 1e100;
        else
            ps_prev = nps{l}(k, prev);
        end
        if(after == -1)
            ps_after = -1e100;
        else
            ps_after = nps{l}(k, after);
        end
        sli = s{l}(i);
        ub = (ps_prev - sli) / delta(l);
        lb = (ps_after - sli) / delta(l);
        nps{l}(k,i) = rtnorm(lb, ub) * delta(l) + sli;
    end
end

function [ps, s, delta] = initialize(data, qnum, wnum, dnum)
    s = cell(1, qnum);
    for l = 1:qnum
        s{l} = rand(dnum(l), 1);
    end
    ps = cell(1, qnum);
    for l = 1:qnum
        ps{l} = rand(wnum, dnum(l));
    end
    for t = length(data):-1:1
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        prev = data(t, 4);
        after = data(t, 5);
        if(after ~= -1)
            ps{l}(k,i) = ps{l}(k, after) + rand();
        end
    end
    delta = ones(qnum, 1) + 1e-2;    
    [ps, s, delta] = normalize(ps, s, delta);
end

function [ps, s, delta] = normalize(ps, s, delta)
    if(false)
        r = sqrt(sum(delta .^ 2));
        delta = delta ./ r * 10;
        for l = 1:length(s)
            shift = min(s{l});
            s{l} = (s{l} - shift) ./ r;
            ps{l} = (ps{l} - shift) ./ r;
        end
    end
end

function check_ps(data, ps)
    fprintf('checking ps\n');
    for t = 1:length(data)
        k = data(t, 1);
        l = data(t, 2);
        i = data(t, 3);
        prev = data(t, 4);
        after = data(t, 5);
        if(after ~= -1)
            if(ps{l}(k,i) < ps{l}(k, after))
                fprintf('l=%d,k=%d, i=%d (%g), after=%d (%g)\n', ...
                        l, k, i, ps{l}(k,i), after, ps{l}(k, after));
                fprintf('error\n');
                pause;
            end
        end
        if(prev ~= -1)
            if(ps{l}(k,i) > ps{l}(k, prev))
                fprintf('l=%d,k=%d, i=%d (%g), prev=%d (%g)\n', ...
                        l, k, i, ps{l}(k,i), prev, ps{l}(k, prev));
                fprintf('error\n');
                pause;
            end
        end
    end
    fprintf('successful\n');
end
