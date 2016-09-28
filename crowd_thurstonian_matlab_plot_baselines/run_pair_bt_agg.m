function run_pair_bt_agg(sr, lambda, beg_sec)
    sr_vec = [0.2, 1.0];
    lambda_vec = [0.05, 0.1, 0.5, 1.0, 5, 10];
    for sec = beg_sec:5
        data_file_path = sprintf(['/home/xwang95/exp/tpp/' ... 
                        'mq2008agg_small/mq2008agg_small_sparse_' ...
                        '%.1f_%d.mat.txt'], sr, sec);
        dn_file_path = ['/home/xwang95/exp/tpp/mq2008agg_small/' ... 
                        'mq2008agg_small_dn.mat.txt'];
        dump_file_path = sprintf(['/home/xwang95/exp/tpp/' ...
                         'mq2008agg_small/mq2008agg_small_sparse_' ...
                         '%.1f_%d_%g'], sr, sec, lambda);
        data = importdata(data_file_path);
        dnum = importdata(dn_file_path);
        wnum = max(data(:, 1));
        qnum = length(dnum);
        %             if(sr == 0.2 && lambda == 0.05)
        %                 continue;
        %             end
        fprintf('lambda = %g, sr = %g, section = %d\n', lambda, sr, sec);
        fprintf('data_file_path: %s\n', data_file_path);
        pair_bt(data, lambda, qnum, wnum, dnum, dump_file_path, 30);
    end
end