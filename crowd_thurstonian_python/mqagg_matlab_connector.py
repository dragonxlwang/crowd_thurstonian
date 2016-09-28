import os;
import crowd_thurstonian
from crowd_thurstonian.mqagg import MQAgg, BordaCount
from toolkit.utility import parseNumVal
from toolkit.num.arithmetic import std, avg, var;
import sys;

def process(sr, iter):
    (mqagg, map_l_id, map_l_i_id, map_id_l, map_l_id_i) = init();
    pairPrefLstFilePath = os.path.join(os.path.expanduser('~'),
                                               'exp/tpp/mq2008agg_small',
               'mq2008agg_small_sparse_{0}_{1}.data'.format(sr, iter));
    mq2008agg_m_file_path = os.path.join(os.path.expanduser('~'),
                                         'exp/tpp/mq2008agg_small',
               'mq2008agg_small_sparse_{0}_{1}.mat.txt'.format(sr, iter));
    mq2008agg_m_file = open(mq2008agg_m_file_path, 'w');
    
    pairPrefList = crowd_thurstonian.mqagg.loadPairPrefLst(pairPrefLstFilePath);
    for (k, l, i, j) in pairPrefList:
        mq2008agg_m_file.write(' '.join([str(x) for x in 
                [k, map_l_id[l], map_l_i_id[l][i], map_l_i_id[l][j]]]) + '\n');
    mq2008agg_m_file.close();
    print mq2008agg_m_file_path;
    return;

def processForTRM():
    (mqagg, map_l_id, map_l_i_id, map_id_l, map_l_id_i) = init();
    mq2008agg_trm_m_file_path = os.path.join(os.path.expanduser('~'),
                                         'exp/tpp/mq2008agg_small',
               'mq2008agg_small_trm.mat.txt');
    mq2008agg_trm_s_init_m_file_path = os.path.join(os.path.expanduser('~'),
                                         'exp/tpp/mq2008agg_small',
               'mq2008agg_small_trm_s_bc_init.mat.txt');
    bc = BordaCount(mqagg.paramPS);
    with open(mq2008agg_trm_m_file_path, "w") as fout:
        for k in mqagg.paramPS:
            for l in mqagg.paramPS[k]:
                    rank_list = sorted(mqagg.paramPS[k][l],
                                       key=lambda i:-mqagg.paramPS[k][l][i]);
                    for t in range(len(rank_list)):
                        prev = -1;
                        after = -1;
                        if(t != 0): 
                            prev = map_l_i_id[l][rank_list[t - 1]];
                        if(t != len(rank_list) - 1):
                            after = map_l_i_id[l][rank_list[t + 1]];
                        d = [k, map_l_id[l], map_l_i_id[l][rank_list[t]],
                             prev, after];
                        fout.write(' '.join([str(x) for x in d]) + '\n');
    with open(mq2008agg_trm_s_init_m_file_path, "w") as fout:
        for l in bc:
            for i in bc[l]:
                fout.write(', '.join(str(x) for x in 
                    [map_l_id[l], map_l_i_id[l][i], bc[l][i]]) + '\n');
    return;
                
    
def init():
    mqagg = MQAgg();
    map_l_id = {};
    map_l_i_id = {};
    map_id_l = {};
    map_l_id_i = {};
    for l in mqagg.paramGS:
        if(l not in map_l_id): 
            map_l_id[l] = 1 + len(map_l_id);
            map_id_l[map_l_id[l]] = l;
            map_l_i_id[l] = {};
            map_l_id_i[l] = {};
        for i in mqagg.paramGS[l]:
            if(i not in map_l_i_id[l]): 
                map_l_i_id[l][i] = 1 + len(map_l_i_id[l]);
                map_l_id_i[l][map_l_i_id[l][i]] = i;    
    file = open(os.path.join(os.path.expanduser('~'),
                                         'exp/tpp/mq2008agg_small',
               'mq2008agg_small_dn.mat.txt'), 'w');    
    for i in range(len(map_id_l)):
        l = map_id_l[1 + i];
        dn = len(map_l_i_id[l]);
        file.write('{0}\n'.format(dn));
    file.close();
    return (mqagg, map_l_id, map_l_i_id, map_id_l, map_l_id_i);
    
def evaluate(lmbda, sr, iter, epoch=30):
    (mqagg, map_l_id, map_l_i_id, map_id_l, map_l_id_i) = init();
    s_csv_file_path = os.path.join(os.path.expanduser('~'),
                                   'exp/tpp/mq2008agg_small',
    'mq2008agg_small_sparse_{0}_{1}_{2}_s_final'.format(sr, iter, lmbda));
    with open(s_csv_file_path, 'r') as f:
        d = [[parseNumVal(x) for x in l.split(',')] for l in  f.readlines()];
    rs = {};
    for [l, i, s] in d:
        if(map_id_l[l] not in rs): rs[map_id_l[l]] = {};
        rs[map_id_l[l]][map_l_id_i[map_id_l[l]][i]] = s;
    ndcg2Lst = [];
    ndcg4Lst = [];
    ndcg6Lst = [];
    ndcg8Lst = [];
    for l in mqagg.paramGS:
        ndcg2Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            2));
        ndcg4Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            4));
        ndcg6Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            6));
        ndcg8Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            8));
    ndcg2Lst = [x for x in ndcg2Lst if(x is not None)];
    ndcg4Lst = [x for x in ndcg4Lst if(x is not None)];
    ndcg6Lst = [x for x in ndcg6Lst if(x is not None)];
    ndcg8Lst = [x for x in ndcg8Lst if(x is not None)];
    print sr, iter;
    print avg(ndcg2Lst), std(ndcg2Lst);
    print avg(ndcg4Lst), std(ndcg4Lst);
    print avg(ndcg6Lst), std(ndcg6Lst);
    print avg(ndcg8Lst), std(ndcg8Lst);
    return [avg(ndcg2Lst), std(ndcg2Lst),
            avg(ndcg4Lst), std(ndcg4Lst),
            avg(ndcg6Lst), std(ndcg6Lst),
            avg(ndcg8Lst), std(ndcg8Lst)];

def evaluateTRM(epoch):
    (mqagg, map_l_id, map_l_i_id, map_id_l, map_l_id_i) = init();
    s_csv_file_path = os.path.join(os.path.expanduser('~'),
                                   'exp/tpp/mq2008agg_small/trm_result/',
                                   's_{0}'.format(epoch));
#     s_csv_file_path = os.path.join(os.path.expanduser('~'),
#                                    'exp/tpp/mq2008agg_small/',
#                                    'mq2008agg_small_trm_s_bc_init.mat.txt');
    with open(s_csv_file_path, 'r') as f:
        d = [[parseNumVal(x) for x in l.split(',')] for l in  f.readlines()];
    rs = {};
    for [l, i, s] in d:
        if(map_id_l[l] not in rs): rs[map_id_l[l]] = {};
        rs[map_id_l[l]][map_l_id_i[map_id_l[l]][i]] = s;
    ndcg2Lst = [];
    ndcg4Lst = [];
    ndcg6Lst = [];
    ndcg8Lst = [];
    for l in mqagg.paramGS:
        ndcg2Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            2));
        ndcg4Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            4));
        ndcg6Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            6));
        ndcg8Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(rs[l],
                                                            mqagg.paramGS[l],
                                                            8));
#     print len(ndcg2Lst), len(ndcg4Lst), len(ndcg6Lst), len(ndcg8Lst);
    
    ndcg2Lst = [x for x in ndcg2Lst if(x is not None)];
    ndcg4Lst = [x for x in ndcg4Lst if(x is not None)];
    ndcg6Lst = [x for x in ndcg6Lst if(x is not None)];
    ndcg8Lst = [x for x in ndcg8Lst if(x is not None)];
    print("results for epoch [{0}]".format(epoch));
    print avg(ndcg2Lst), std(ndcg2Lst);
    print avg(ndcg4Lst), std(ndcg4Lst);
    print avg(ndcg6Lst), std(ndcg6Lst);
    print avg(ndcg8Lst), std(ndcg8Lst);
#     print len(ndcg2Lst), len(ndcg4Lst), len(ndcg6Lst), len(ndcg8Lst);
    return [avg(ndcg2Lst), std(ndcg2Lst),
            avg(ndcg4Lst), std(ndcg4Lst),
            avg(ndcg6Lst), std(ndcg6Lst),
            avg(ndcg8Lst), std(ndcg8Lst)];
                
def previous_main_for_bt():
#     process(0.2, 1);
#     process(0.2, 2);
#     process(0.2, 3);
#     process(0.2, 4);
#     process(0.2, 5);
#     process(1.0, 1);
#     process(1.0, 2);
#     process(1.0, 3);
#     process(1.0, 4);
#     process(1.0, 5);
    sr = 1.0;
    ndcg2lst = [];
    ndcg4lst = [];
    ndcg6lst = [];
    ndcg8lst = [];
    lmbda_vec = [0.05, 0.1, 0.5, 1, 5, 10];
    lmbda = 100000;
    for iter in range(1, 6):
        arr = evaluate(lmbda, sr, iter);
        ndcg2lst.append(arr[0]);
        ndcg4lst.append(arr[2]);
        ndcg6lst.append(arr[4]);
        ndcg8lst.append(arr[6]);
    print 80 * '~';
    print('sr = {0}'.format(sr));
    print('lambda = {0}'.format(lmbda));
    print avg(ndcg2lst), std(ndcg2lst);
    print avg(ndcg4lst), std(ndcg4lst);
    print avg(ndcg6lst), std(ndcg6lst);
    print avg(ndcg8lst), std(ndcg8lst);
    return;

if(__name__ == "__main__"):
    beg = 101;
    end = 200;
    l = [[] for i in range(end - beg + 1)];
    for iter in range(beg, end + 1):
        x = evaluateTRM(iter);
        l[0].append(x[0]);
        l[1].append(x[2]);
        l[2].append(x[4]);
        l[3].append(x[6]);
        print("");
    print "STD";
    print std(l[0]);
    print std(l[1]);
    print std(l[2]);
    print std(l[3]);
#     processForTRM();
