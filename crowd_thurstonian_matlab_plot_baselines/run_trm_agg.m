function run_trm_agg()
%RUN_TRM_AGG Summary of this function goes here
%   Detailed explanation goes here
    mq2008agg_trm_m_file_path =  ['/home/xwang95/exp/' ...
                    'tpp/mq2008agg_small/mq2008agg_small_trm.mat.txt'];
    dn_file_path = ['/home/xwang95/exp/tpp/mq2008agg_small/' ... 
                    'mq2008agg_small_dn.mat.txt'];
    dump_file_path = '/home/xwang95/exp/tpp/mq2008agg_small/trm_result/';
    data = importdata(mq2008agg_trm_m_file_path);
    dnum = importdata(dn_file_path);
    wnum = max(data(:, 1));
    qnum = length(dnum);
    trm(data, qnum, wnum, dnum, dump_file_path, 100);
end