'''
Created on Jan 18, 2014

@author: xwang95
'''

import re;
import os;
from toolkit.utility import parseNumVal
from toolkit.num.arithmetic import avg, std, var;
from crowd_thurstonian.simulator import TPPDataSimulator;
import itertools;
iterRe = re.compile('Iter = (\d+)');
elboRe = re.compile('ELBO: ([\d.+-]+e?[\d.+-]+)');
maliciousRe = re.compile('MALICIOUS: prec = ([\d.+-]+e?[\d.+-]+); recall = ([\d.+-]+e?[\d.+-]+); f1 = ([\d.+-]+e?[\d.+-]+)');
truthRe = re.compile('MALICIOUS: prec = ([\d.+-]+e?[\d.+-]+); recall = ([\d.+-]+e?[\d.+-]+); f1 = ([\d.+-]+e?[\d.+-]+)')
ktRe = re.compile('avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');

def processFile(filePath):
    iter = -1;
    val = {};
    fin = open(filePath, 'r');
    for ln in fin:
        m = iterRe.search(ln);
        if(m): 
            iter = parseNumVal(m.group(1));
            val[iter] = {};            
            continue;
        m = elboRe.search(ln);
        if(m):
            elbo = parseNumVal(m.group(1));
            val[iter]['elbo'] = elbo;
            continue;
        m = ktRe.search(ln);
        if(m):
            (ktavg, ktstd) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['avg'] = ktavg;
            val[iter]['std'] = ktstd;
    if(('avg' not in val[iter]) or ('std' not in val[iter])): del val[iter];
    fin.close();
    return val;

def collectEvalFiles(filePathPatt):
    valLst = {};
    maxIter = 99999;
    retVal = {};
    for i in range(1, 6):
        filePath = filePathPatt.format(i);
        valLst[i] = processFile(filePath);
        if(len(valLst[i]) < maxIter): maxIter = len(valLst[i]);
#         print(len(valLst[i]));
#     print('');
    for iter in range(maxIter):
        elboLst = [valLst[i][iter]['elbo'] for i in range(1, 6)];
        avgLst = [valLst[i][iter]['avg'] for i in range(1, 6)];
        stdLst = [valLst[i][iter]['std'] for i in range(1, 6)];
        retVal[iter] = [avg(elboLst), std(elboLst), avg(avgLst), std(avgLst), avg(stdLst), std(stdLst)];
        print('{0} {1} {2} {3} {4} {5} {6}'.format(iter, avg(elboLst), std(elboLst), avg(avgLst), std(avgLst), avg(stdLst), std(stdLst)));

ndcg2Re = re.compile('NDCG@2: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
ndcg4Re = re.compile('NDCG@4: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
ndcg6Re = re.compile('NDCG@6: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
ndcg8Re = re.compile('NDCG@8: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
def processMQ2008File(filePath):
    iter = -1;
    val = {};
    fin = open(filePath, 'r');
    for ln in fin:
        m = iterRe.search(ln);
        if(m): 
            iter = parseNumVal(m.group(1));
            val[iter] = {};            
            continue;
        m = elboRe.search(ln);
        if(m):
            elbo = parseNumVal(m.group(1));
            val[iter]['elbo'] = elbo;
            continue;
        m = ndcg2Re.search(ln);
        if(m):
            (a, s) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['ndcg2avg'] = a;
            val[iter]['ndcg2std'] = s;
        m = ndcg4Re.search(ln);
        if(m):
            (a, s) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['ndcg4avg'] = a;
            val[iter]['ndcg4std'] = s;
        m = ndcg6Re.search(ln);
        if(m):
            (a, s) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['ndcg6avg'] = a;
            val[iter]['ndcg6std'] = s;
        m = ndcg8Re.search(ln);
        if(m):
            (a, s) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['ndcg8avg'] = a;
            val[iter]['ndcg8std'] = s;
    if(('ndcg2avg' not in val[iter]) or
       ('ndcg4avg' not in val[iter]) or
       ('ndcg6avg' not in val[iter]) or
       ('ndcg8avg' not in val[iter]) or 
       ('ndcg2std' not in val[iter]) or
       ('ndcg4std' not in val[iter]) or 
       ('ndcg6std' not in val[iter]) or
       ('ndcg8std' not in val[iter])): del val[iter];
    fin.close();
    return val;

def collectMQ2008EvalFiles(filePathPatt):
    valLst = {};
    maxIter = 99999;
    for i in range(1, 6):
        filePath = filePathPatt.format(i);
        valLst[i] = processMQ2008File(filePath);
        if(len(valLst[i]) < maxIter): maxIter = len(valLst[i]);
#         print(len(valLst[i]));
#     print('');
    for iter in range(maxIter):
        elboLst = [valLst[i][iter]['elbo'] for i in range(1, 6)];
        ndcg2avgLst = [valLst[i][iter]['ndcg2avg'] for i in range(1, 6)];
        ndcg2stdLst = [valLst[i][iter]['ndcg2std'] for i in range(1, 6)];
        ndcg4avgLst = [valLst[i][iter]['ndcg4avg'] for i in range(1, 6)];
        ndcg4stdLst = [valLst[i][iter]['ndcg4std'] for i in range(1, 6)];
        ndcg6avgLst = [valLst[i][iter]['ndcg6avg'] for i in range(1, 6)];
        ndcg6stdLst = [valLst[i][iter]['ndcg6std'] for i in range(1, 6)];
        ndcg8avgLst = [valLst[i][iter]['ndcg8avg'] for i in range(1, 6)];
        ndcg8stdLst = [valLst[i][iter]['ndcg8std'] for i in range(1, 6)];
        print('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18}'.format(iter, avg(elboLst), std(elboLst),
                                                   avg(ndcg2avgLst), std(ndcg2avgLst), avg(ndcg2stdLst), std(ndcg2stdLst),
                                                   avg(ndcg4avgLst), std(ndcg4avgLst), avg(ndcg4stdLst), std(ndcg4stdLst),
                                                   avg(ndcg6avgLst), std(ndcg6avgLst), avg(ndcg6stdLst), std(ndcg6stdLst),
                                                   avg(ndcg8avgLst), std(ndcg8avgLst), avg(ndcg8stdLst), std(ndcg8stdLst)));

def processRecordExp1():
    string = '''
0.5028    0.4992    0.4984    0.4956    0.5292    0.5596    0.806    1    0.98    0.98    0.98
0.514    0.5352    0.532    0.542    0.602    0.6708    0.768    0.9624    0.98    0.98    0.98
                                        
0.4948    0.496    0.4816    0.4812    0.496    0.52    0.6636    0.8696    0.8896    0.92    0.92
0.4868    0.5176    0.5232    0.5196    0.5256    0.526    0.5784    0.8296    0.98    0.98    0.98
                                        
0.5088    0.5092    0.5232    0.5176    0.534    0.552    0.6552    0.9632    0.866    0.88    0.88
0.4996    0.4856    0.4964    0.5008    0.528    0.5192    0.5932    0.848    0.98    1    1
                                        
0.5076    0.5136    0.5116    0.5352    0.5424    0.5968    0.8808    0.9884    0.98    0.98    0.98
0.504    0.5132    0.5056    0.54    0.5752    0.6384    0.8112    0.9412    0.94    0.94    0.94
                                        
0.4876    0.4884    0.514    0.5204    0.54    0.6468    0.9044    1    1    1    1
0.488    0.486    0.4576    0.4628    0.4392    0.428    0.5536    0.8944    0.9    0.9    0.9
                                        
0.5044    0.4944    0.4924    0.4884    0.5376    0.7308    0.9908    0.76    0.76    0.76    0.76
0.5008    0.516    0.5348    0.53    0.5576    0.672    0.91    1    1    1    1
                                        
0.5012    0.5112    0.5156    0.512    0.5012    0.5408    0.6316    0.8272    0.74    0.74    0.74
0.5024    0.4992    0.488    0.5036    0.502    0.4924    0.4984    0.6996    0.8    0.8    0.8
                                        
0.496    0.488    0.4988    0.5152    0.5252    0.532    0.494    0.352    0.6176    0.68    0.68
0.5024    0.5056    0.522    0.5168    0.5176    0.5204    0.5196    0.644    0.7344    0.96    0.96
                                        
0.5304    0.5228    0.5424    0.5656    0.592    0.7348    0.9736    0.6952    0.72    0.72    0.72
0.5056    0.472    0.4832    0.4832    0.4928    0.5692    0.8476    1    1    1    1

0.4944    0.504    0.49    0.4968    0.5232    0.5836    0.7744    0.9916    0.7128    0.74    0.74
0.4964    0.4988    0.4952    0.5128    0.504    0.522    0.5768    0.892    1    1    1
''';
    lnLst = [ln for ln in string.split('\n') if (ln.strip())];
    data = [[] for i in range(10)];
    for i in range(10):
        data[i].append([parseNumVal(x) for x in lnLst[i * 2].split()]);
        data[i].append([parseNumVal(x) for x in lnLst[i * 2 + 1].split()]);
    acc = {};
    f1pos = {};
    f1neg = {};
    f1 = {};
    for iter in range(10):
            acc[iter] = [(data[i][0][iter] + data[i][1][iter]) / 2 for i in range(10)];
            f1pos[iter] = [  2.0 / (1.0 / (data[i][0][iter] / (data[i][0][iter] + 1 - data[i][1][iter])) + 1.0 / (data[i][0][iter]))  for i in range(10) ]
            f1neg[iter] = [  2.0 / (1.0 / (data[i][1][iter] / (data[i][1][iter] + 1 - data[i][0][iter])) + 1.0 / (data[i][1][iter]))  for i in range(10) ]
            f1[iter] = [(f1neg[iter][i] + f1pos[iter][i]) / 2.0 for i in range(10)];
            print iter;
            print avg(acc[iter]), std(acc[iter]);
            print avg(f1pos[iter]), std(f1pos[iter]);
            print avg(f1neg[iter]), std(f1neg[iter]); 
            print avg(f1[iter]), std(f1[iter]);
            
def matchDomainByF1(predParamTau, genParamTau, dNum):
    permIter = itertools.permutations(range(dNum));
    bestPerm = None;
    bestF1 = -9999999;
    bestPrec = 0;
    bestRecall = 0;
    for perm in permIter:
        (tp, fp, fn, tn) = (0.0, 0.0, 0.0, 0.0);
        for k in genParamTau:
            for m in genParamTau[k]:
                g = genParamTau[k][m];
                t = predParamTau[k][perm[m]];
                if(g < 0.0 and t < 0.0): tp += 1.0;
                elif(g < 0.0 and t > 0.0): fn += 1.0;
                elif(g > 0.0 and t < 0.0): fp += 1.0;
                else: tn += 1.0
        if(tp + fp == 0.0): prec = 1.0;
        else: prec = tp / (tp + fp);
        if(tp + fn == 0.0): recall = 1.0;
        else: recall = tp / (tp + fn);
        f1 = 2 * prec * recall / (prec + recall + (1e-10));
        if(f1 > bestF1):
            bestF1 = f1;
            bestPerm = perm;
            bestPrec = prec;
            bestRecall = recall; 
    return (bestPerm, bestF1, bestPrec, bestRecall);

def rocCurve(modelFilePath, predModelFilePathPatt, showIterF1=False, rocIter=None):
    dNum = 5;
    sim = TPPDataSimulator();
    sim.loadModel(modelFilePath);
    genParamTau = sim.paramTau;
    
    recLst = {};
    predParamTau = {};
    
    maxIter = 0;
    while(True):
        stop = False;
        for i in range(1, 6):
            if(not os.path.exists(predModelFilePathPatt.format(i, maxIter))):
                stop = True;
                break;
        if(stop): break;
        maxIter += 1;
    maxIter -= 1;
    
    if(showIterF1):
        print 'maxIter = {0}'.format(maxIter);
#         for iter in range(maxIter + 1):
        for iter in range(10000):
            f1Lst = [];
            precLst = [];
            recallLst = [];
            permLst = [];
            stop = 0;
            for i in range(1, 6):
                predModelFilePath = predModelFilePathPatt.format(i, iter);
                if(not os.path.exists(predModelFilePath)):
                    stop += 1;
                    continue;
                pred = TPPDataSimulator();
                pred.loadModel(predModelFilePath);
                predParamTau = pred.paramTau;
                (bestPerm, bestF1, bestPrec, bestRecall) = matchDomainByF1(predParamTau, genParamTau, dNum);
                f1Lst.append(bestF1);
                precLst.append(bestPrec);
                recallLst.append(bestRecall);
                permLst.append(bestPerm);
            if(stop == 5): return;
            print('iter = {0}, F1 = {1} {2}, prec = {3} {4}  recall = {5} {6} perm1 = {7}'.format(iter, avg(f1Lst), std(f1Lst),
                                                                                                        avg(precLst), std(precLst),
                                                                                                        avg(recallLst), std(recallLst),
                                                                                                         permLst[0]));
        return;

    if(rocIter is not None):
        combinedParamTau = {};
        (tp, fp, fn, tn) = (0.0, 0.0, 0.0, 0.0);
        for i in range(1, 6):
            predModelFilePath = predModelFilePathPatt.format(i, rocIter);
            pred = TPPDataSimulator();
            pred.loadModel(predModelFilePath);
            predParamTau = pred.paramTau;
            (bestPerm, bestF1, bestPrec, bestRecall) = matchDomainByF1(predParamTau, genParamTau, dNum);
            for k in genParamTau:
                for m in genParamTau[k]:
                    if(k not in combinedParamTau): combinedParamTau[k] = {};
                    if(m not in combinedParamTau[k]): combinedParamTau[k][m] = 0.0;
                    combinedParamTau[k][m] += predParamTau[k][bestPerm[m]] / 5.0;
        for k in combinedParamTau:
            for m in combinedParamTau[k]:
                recLst[(k, m, genParamTau[k][m], combinedParamTau[k][m])] = combinedParamTau[k][m];
        roc = [];
        for (k, m, gsTau, mdTau) in recLst:
            if(gsTau < 0.0): fn += 1.0;
            else: tn += 1.0;
        roc.append([fp / (fp + tn), tp / (tp + fn)]);
        for (k, m, gsTau, mdTau) in sorted(recLst, key=lambda x: recLst[x]):
            if(gsTau < 0.0): 
                tp += 1.0;
                fn -= 1.0;
            else:
                fp += 1.0;
                tn -= 1.0;
            roc.append([fp / (fp + tn), tp / (tp + fn)]);
        print '[', ;
        for l in range(len(roc)):
            if(l != len(roc) - 1):  print roc[l], ';', ;
            else: print roc[l], ;
        print('];');
        auc = 0.0;
        for l in range(len(roc) - 1): auc += (roc[l + 1][0] - roc[l][0]) * (roc[l + 1][1] + roc[l][1]) / 2.0;
        print 'auc = {0}'.format(auc);
              
   

def processLargeAnnoFile(filePath):
    iter = -1;
    val = {};
    fin = open(filePath, 'r');
    for ln in fin:
        m = iterRe.search(ln);
        if(m): 
            iter = parseNumVal(m.group(1));
            val[iter] = {};            
            continue;
        m = elboRe.search(ln);
        if(m):
            elbo = parseNumVal(m.group(1));
            val[iter]['elbo'] = elbo;
            continue;
        m = maliciousRe.search(ln);
        if(m):
            (prec, recall) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['malicious_prec'] = prec;
            val[iter]['malicious_recall'] = recall;
        m = truthRe.search(ln);
        if(m):
            (prec, recall) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['truth_prec'] = prec;
            val[iter]['truth_recall'] = recall;
        m = ktRe.search(ln);
        if(m):
            (ktavg, ktstd) = (parseNumVal(m.group(1)), parseNumVal(m.group(2)));
            val[iter]['avg'] = ktavg;
            val[iter]['std'] = ktstd;
    if(('elbo' not in val[iter]) or 
       ('malicious_prec' not in val[iter]) or
       ('malicious_recall' not in val[iter]) or
       ('truth_prec' not in val[iter]) or
       ('truth_recall' not in val[iter])): del val[iter];
    fin.close();
    return val;

def collectLargeAnnoFiles(filePathPatt):
    valLst = {};
    maxIter = 99999;
    retVal = {};
    for i in range(1, 6):
        filePath = filePathPatt.format(i);
        valLst[i] = processLargeAnnoFile(filePath);
        if(len(valLst[i]) < maxIter): maxIter = len(valLst[i]);
#         print(len(valLst[i]));
#     print('');
    for iter in range(maxIter):
        elboLst = [valLst[i][iter]['elbo'] for i in range(1, 6)];
        maliciousPrecLst = [valLst[i][iter]['malicious_prec'] for i in range(1, 6)];
        maliciousRecallLst = [valLst[i][iter]['malicious_recall'] for i in range(1, 6)];
        truthPrecLst = [valLst[i][iter]['truth_prec'] for i in range(1, 6)];
        truthRecallLst = [valLst[i][iter]['truth_recall'] for i in range(1, 6)];
        avgLst = [valLst[i][iter]['avg'] for i in range(1, 6)];
        stdLst = [valLst[i][iter]['std'] for i in range(1, 6)];
        print('{0} {1} {2} {3} {4} {5} {6} {7}'.format(iter, avg(avgLst), std(avgLst), avg(elboLst), avg(maliciousPrecLst), avg(maliciousRecallLst), avg(truthPrecLst), avg(truthRecallLst))
              );

ndcg2Re = re.compile('NDCG@2: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
ndcg4Re = re.compile('NDCG@4: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
ndcg6Re = re.compile('NDCG@6: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
ndcg8Re = re.compile('NDCG@8: avg = ([\d.+-]+e?[\d.+-]+); std = ([\d.+-]+e?[\d.+-]+)');
    
if __name__ == '__main__':
    # 100 anno 0.01 sparse 
    # iter = 54, F1 = 0.497832117507 0.0377647455791, prec = 0.826025290499 0.110522508371  recall = 0.36 0.0382325567424 perm1 = (3, 1, 0, 4, 2)
    rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_100anno.model'),
             predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_100anno_sparse_0.01_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
             showIterF1=False, rocIter=54);        
#     rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_100anno.model'),
#              predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_100anno_sparse_0.01_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
#              showIterF1=True, rocIter=None);        
 
 
    # 100 anno 0.02 sparse
    # iter = 42, F1 = 0.443055218708 0.131387638048, prec = 0.496125116713 0.129877879738  recall = 0.404444444444 0.133629301141 perm1 = (4, 1, 0, 3, 2)
    rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_100anno.model'),
             predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_100anno_sparse_0.02_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
             showIterF1=False, rocIter=42);
#     rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_100anno.model'),
#            predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_100anno_sparse_0.02_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
#            showIterF1=True, rocIter=None);         


    # 200 anno 0.01 sparse
    # iter = 57, F1 = 0.584064620725 0.134649982621, prec = 0.842791913135 0.0693731844272  recall = 0.45504587156 0.136398293872 perm1 = (3, 0, 1, 4, 2)
    # iter = 60, F1 = 0.637066389455 0.0556710406646, prec = 0.839682539683 0.0460317460317  recall = 0.51376146789 0.0550458715596 perm1 = (0, 1, 2, 4, 3)
    rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_200anno.model'),
             predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_200anno_sparse_0.01_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
             showIterF1=False, rocIter=57);
#     rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_200anno.model'),
#              predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_200anno_sparse_0.01_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
#              showIterF1=True, rocIter=None);


#     # 200 anno 0.02 sparse
#     # iter = 23, F1 = 0.453645289708, prec = 0.663729842562, recall = 0.363302752294, perm1 = (0, 4, 3, 1, 2)
    rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_200anno.model'),
             predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_200anno_sparse_0.02_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
             showIterF1=False, rocIter=23);
#     rocCurve(modelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_200anno.model'),
#              predModelFilePathPatt=os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_200anno_sparse_0.02_{0}_burnin_20_sample_20_iter_{1}.model.estimate'),
#             showIterF1=True, rocIter=None);    


#     processRecordExp1();
#     collectEvalFiles(os.path.join(os.path.expanduser('~'), 'terminal_ori_sparse_0.1_{0}_20_20'));
    
#     collectMQ2008EvalFiles(os.path.join(os.path.expanduser('~'), 'mq2008_small_sparse_0.6_domain_5_{0}_30_30'));
        
    pass;
