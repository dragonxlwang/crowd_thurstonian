function run_pair_bt_sim(data_id, dnum)
    kt_vec = zeros(5, 1);
    for sec = 1:5
        kt_vec(sec) = run_pair_bt_sim_per_sec(data_id, sec, dnum);
    end
    fprintf('--------------------------------\n');
    fprintf('kt_avg = %f, kt_std = %f\n', mean(kt_vec), std(kt_vec));
end

function kt = run_pair_bt_sim_per_sec(data_id, sec_id, dnum)
    data_vec = {
        'model_gen_short_%d.data', ...
        'model_gen_short_sparse_0.5_%d.data', ...
        'model_gen_short_sparse_malicious_%d.data', ...
        'model_gen_short_sparse_spammer_%d.data', ...
        'model_gen_ori_sparse_0.1_%d.data', ...
        'model_gen_ori_sparse_0.05_%d.data', ...
        'model_gen_ori_sparse_0.05_malicious_%d.data', ...
        'model_gen_ori_sparse_0.05_spammer_%d.data'
        };
    data_file_path = ['/home/xwang95/exp/tpp/exp3/', ...
                      sprintf(data_vec{data_id}, sec_id), ...
                      '.csv'];
    fprintf('data_file_path = %s\n', data_file_path);
    data = importdata(data_file_path);
    new_data = [];
    if(true)
        [m, n] = size(data);
        for i = 1:m
            if(data(i,1) <= 10)
                new_data = [new_data; data(i,:)];
            end
        end
        data = new_data;
    end
    
    lambda = 0.5;
    qnum = 100;
    wnum = 10;
    dnum_vec = dnum * ones(qnum, 1);
    dump_file_path = ['/home/xwang95/exp/tpp/exp3/', ...
                     sprintf(data_vec{data_id}, sec_id), ...
                     sprintf('_%g', lambda)
                     ];
    [s, eta, f] = pair_bt(data, lambda, qnum, wnum, dnum_vec, ...
                        dump_file_path, 30);
    kt_vec = zeros(qnum, 1);
    for l = 1:qnum
        kt_vec(l) = eval_kendall_tau(s{l});       
    end
    fprintf('kt_avg = %f, kt_std = %f\n', mean(kt_vec), std(kt_vec));
    kt = mean(kt_vec);
end

function kt = eval_kendall_tau(gs)
    kt = 0;
    docNum = length(gs);
    for i = 1:docNum
        for j = (i + 1):docNum
            if(gs(i) > gs(j))
                kt = kt + 1;
            end
        end
    end
end