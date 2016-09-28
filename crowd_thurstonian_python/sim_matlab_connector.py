'''
Created on Nov 13, 2014

@author: xwang95
'''
import os;
import crowd_thurstonian.mqagg;

def process():
    exp_dir = '/home/xwang95/exp/tpp/exp3/';
    fp_lst = [exp_dir+fp for fp in os.listdir(exp_dir) 
              if(fp.endswith(".data"))];
    for fp in fp_lst:
        pplst = crowd_thurstonian.mqagg.loadPairPrefLst(fp);
        print fp;
        with open(fp+'.csv', 'w') as f:
            for (k, l, i, j) in pplst:
                f.write(', '.join(str(x) for x in [k+1, l+1, i+1, j+1]) + '\n');
    return;

if __name__ == '__main__':
    process();
    pass