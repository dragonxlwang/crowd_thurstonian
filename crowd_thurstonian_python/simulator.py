'''
Created on Jan 15, 2014

@author: xwang95
'''
import random;
from toolkit.num.arithmetic import getRandomSubsetIdx;
from toolkit.num.probability import multinomialSampling;
from toolkit.num.arithmetic import getQuotientReminder;
import math
import os

class TPPDataSimulator(object):
    '''
    classdocs
    '''
    # constant
    constWorkerNum = 0;
    constQueryNum = 0;
    constDomainNum = 0;
    
    # data: D
    pairPrefList = [];
    
    # model parameters: Theta
    paramGoldScores = {};  # gold standard scores, index: query, doc; l => i -> score
    paramDeltaSquare = {};  # query difficulty, index: query; l -> delta
    paramTau = {};  # worker param, index: worker; k => m -> tau
    paramTheta = [];  # domain prior, index: query; m -> theta
    
    # latent variables: Z
    latentQueryDomain = {};  # query domain, index: query -> domain; l => m
    latentPerceivedScores = {};  # perceived scores, index: worker, query, doc -> ps; k => l => i -> score
    
    def __init__(self):
        return;
    
    def __sgn(self, x): return 1.0 if x >= 0.0 else -1.0;
    
    def generateModel(self,
                 wNum=10,
                 qNum=100,
                 dNum=2,
                 docNum=20,
                 difficultDocIntNum=5,
                 difficultQueryNum=5,
                 spammerWorkerNum=2,
                 maliciousWorkerNum=1):
        '''
        Constructor
        '''
        self.constWorkerNum = wNum;
        self.constQueryNum = qNum;
        self.constDomainNum = dNum;
        
        
        # model parameters
        # gs
        #    deterministic interval: easy = 10, difficult = 5
        self.paramGoldScores = {};
        for l in range(qNum): 
            self.paramGoldScores[l] = {};
            for i in range(docNum): self.paramGoldScores[l][i] = 0.0;
        for l in range(qNum):
            difficultDocIntSet = set(getRandomSubsetIdx(docNum - 1, difficultDocIntNum));
            for i in range(docNum):
                if(i == 0): self.paramGoldScores[l][i] = 0.0;
                elif(i - 1 in difficultDocIntSet): self.paramGoldScores[l][i] = self.paramGoldScores[l][i - 1] + 5;
                else: self.paramGoldScores[l][i] = self.paramGoldScores[l][i - 1] + 10;
                
        # delta
        #    deterministic theta ratio: easy = 1, difficult = 4
        difficultQuerySet = set(getRandomSubsetIdx(qNum, difficultQueryNum));
        self.paramDeltaSquare = {};  # initialization
        for l in range(qNum): self.paramDeltaSquare[l] = 1; 
        for l in difficultQuerySet: self.paramDeltaSquare[l] = 4 ** 2;
        scaleSquare = sum(self.paramDeltaSquare.values()) / len(self.paramDeltaSquare);  # scaling
        for l in range(qNum): self.paramDeltaSquare[l] /= scaleSquare;
        
        
        #  tau
        #    deterministic tau: 
        poorWorkerLst = getRandomSubsetIdx(wNum, maliciousWorkerNum + spammerWorkerNum);
        maliciousWorkerSet = set(poorWorkerLst[0:maliciousWorkerNum]);
        spammerWorkerSet = set(poorWorkerLst[maliciousWorkerNum:]);
        self.paramTau = {};  # initialization
        for k in range(wNum):
            self.paramTau[k] = {};
            for m in range(dNum):
                self.paramTau[k][m] = 1 + abs(random.gauss(0, 1) * 2);
        for k in maliciousWorkerSet:
            for m in range(dNum): self.paramTau[k][m] = -self.paramTau[k][m];
        for k in spammerWorkerSet:
            for m in range(dNum): self.paramTau[k][m] = 0.01;
            
        # theta
        #    
        self.paramTheta = [random.random() for m in range(dNum)];
        s = sum(self.paramTheta);
        for m in range(dNum): self.paramTheta[m] = self.paramTheta[m] / s;
        
        # rescale
        bias = {};
        for l in self.paramGoldScores:
            gsMin = 1e100;
            gsMax = -1e100;
            for i in self.paramGoldScores[l]:
                if(self.paramGoldScores[l][i] < gsMin): gsMin = self.paramGoldScores[l][i];
                if(self.paramGoldScores[l][i] > gsMax): gsMax = self.paramGoldScores[l][i];
            bias[l] = gsMin;
        
        s = 0.0;
        for l in self.paramDeltaSquare: s += self.paramDeltaSquare[l];
        scale = math.sqrt(s / len(self.paramDeltaSquare));
        
        for l in self.paramGoldScores:  # param: gs
            for i in self.paramGoldScores[l]:
                self.paramGoldScores[l][i] = (self.paramGoldScores[l][i] - bias[l]) / scale;
        for l in self.paramDeltaSquare: self.paramDeltaSquare[l] /= (scale ** 2);  # param: delta
        for k in self.paramTau:  # param: tau
            for m in self.paramTau[k]: self.paramTau[k][m] *= scale;
        return;
    
    def generateModelExp1(self):
        self.constWorkerNum = 5;
        self.constDomainNum = 2;
        self.constQueryNum = 100;
        docNum = 5;
        
        self.paramGoldScores = {};
        for l in range(self.constQueryNum):
            for i in range(docNum):
                if(l not in self.paramGoldScores): self.paramGoldScores[l] = {};
                self.paramGoldScores[l][i] = i * 2.5;
        
        self.paramDeltaSquare = {};
        for l in range(self.constQueryNum): self.paramDeltaSquare[l] = 0.01;
        
        self.paramTau = {};
        for k in range(self.constWorkerNum): self.paramTau[k] = {};
        self.paramTau[0][0] = 1;
        self.paramTau[0][1] = 0.1;
        self.paramTau[1][0] = 0.1;
        self.paramTau[1][1] = 1;
        self.paramTau[2][0] = 10;
        self.paramTau[2][1] = 1;
        self.paramTau[3][0] = 1;
        self.paramTau[3][1] = 10;
        self.paramTau[4][0] = -10;
        self.paramTau[4][1] = 0.1;
        
        self.paramTheta = [0.5, 0.5];        
        return;
    
    docNum = 0;
    demographicPmf = [];
    def generateModelExp3(self, docNum=30, demographicPmf=[0.2, 0.6, 0.1, 0.1], wNum=50, dNum=5):
#         self.constWorkerNum = 10;
        self.constWorkerNum = wNum;
        self.constDomainNum = dNum;
        self.constQueryNum = 100;
        self.docNum = docNum;
        self.demographicPmf = demographicPmf;        
        
        self.paramGoldScores = {};
        for l in range(self.constQueryNum):
            for i in range(self.docNum):
                if(l not in self.paramGoldScores): self.paramGoldScores[l] = {};
                if(i == 0): self.paramGoldScores[l][i] = 0.0;
                else: self.paramGoldScores[l][i] = self.paramGoldScores[l][i - 1] + random.random();
            
        self.paramDeltaSquare = {};
        for l in range(self.constQueryNum): self.paramDeltaSquare[l] = 0.1 * random.random();
        
        self.paramTau = {};
        for k in range(self.constWorkerNum):
            for m in range(self.constDomainNum):
                if(k not in self.paramTau): self.paramTau[k] = {};
                type = multinomialSampling(self.demographicPmf);
                if(type == 0): self.paramTau[k][m] = 10.0;  #  expert
                elif(type == 1): self.paramTau[k][m] = 5.0;  #  normal
                elif(type == 2): self.paramTau[k][m] = 0.1;  # spammer
                elif(type == 3): self.paramTau[k][m] = -10.0;  # malicious
        
        self.paramTheta = [1.0 / self.constDomainNum for m in range(self.constDomainNum)];
        
    def generateData(self, docNum, latentQueryDomain=None, ppqLoad=5):
        if(latentQueryDomain is not None): self.latentQueryDomain = latentQueryDomain;  # latent query domain deterministically generated
        else:  # stochastically generated
            self.latentQueryDomain = {};
            pmf = [self.paramTheta[m] for m in range(len(self.paramTheta))];
            for l in self.paramGoldScores: self.latentQueryDomain[l] = multinomialSampling(pmf);
        
        self.latentPerceivedScores = {};
        for l in self.paramGoldScores:
            for i in self.paramGoldScores[l]:
                for k in self.paramTau: 
                    if(k not in self.latentPerceivedScores): self.latentPerceivedScores[k] = {};
                    if(l not in self.latentPerceivedScores[k]): self.latentPerceivedScores[k][l] = {};
                    self.latentPerceivedScores[k][l][i] = random.gauss(self.paramGoldScores[l][i], math.sqrt(self.paramDeltaSquare[l]));
        
        self.pairPrefList = [];
        i1i2Lst = [];
        for i1 in range(docNum):
            for i2 in range(i1 + 1, docNum):
                i1i2Lst.append((i1, i2));
        for k in self.latentPerceivedScores:
            for l in self.latentPerceivedScores[k]:
                m = self.latentQueryDomain[l];
                for (i1, i2) in [i1i2Lst[x] for x in getRandomSubsetIdx(docNum * (docNum - 1) / 2, ppqLoad)]:
                    std = math.sqrt(2) / abs(self.paramTau[k][m]);
                    mean = self.__sgn(self.paramTau[k][m]) * (self.latentPerceivedScores[k][l][i1] - self.latentPerceivedScores[k][l][i2]);
                    nsdiff = random.gauss(mean, std);
                    if(nsdiff > 0): self.pairPrefList.append((k, l, i1, i2));  # i1 > i2
                    else: self.pairPrefList.append((k, l, i2, i1));  # i2 > i1
        return;
    
    def clear(self):
        # constant
        self.constWorkerNum = 0;
        self.constQueryNum = 0;
        self.constDomainNum = 0;
        
        # data: D
        self.pairPrefList = [];
        
        # model parameters: Theta
        self.paramGoldScores = {};  # gold standard scores, index: query, doc; l => i -> score
        self.paramDeltaSquare = {};  # query difficulty, index: query; l -> delta
        self.paramTau = {};  # worker param, index: worker; k => m -> tau
        self.paramTheta = [];  # domain prior, index: query; m -> theta
        
        # latent variables: Z
        self.latentQueryDomain = {};  # query domain, index: query -> domain; l => m
        self.latentPerceivedScores = {};  # perceived scores, index: worker, query, doc -> ps; k => l => i -> score
        return;
    
    def dumpModel(self, modelFilePath):
        fout = open(modelFilePath, 'w');
        fout.write(str(self.paramGoldScores) + '\n');
        fout.write(str(self.paramDeltaSquare) + '\n');
        fout.write(str(self.paramTau) + '\n');
        fout.write(str(self.paramTheta) + '\n');
        fout.close();
    
    def loadModel(self, modelFilePath):
        fin = open(modelFilePath, 'r');
        self.paramGoldScores = eval(fin.readline().strip());
        self.paramDeltaSquare = eval(fin.readline().strip());
        self.paramTau = eval(fin.readline().strip());
        self.paramTheta = eval(fin.readline().strip());
        fin.close();
        return (self.paramGoldScores, self.paramDeltaSquare, self.paramTau, self.paramTheta);
    
    def dumpLatentVar(self, latentVarFilePath):
        fout = open(latentVarFilePath, 'w');
        fout.write(str(self.latentQueryDomain) + '\n');
        fout.write(str(self.latentPerceivedScores) + '\n');
        fout.close();
        return;
        
    def loadLatentVar(self, latentVarFilePath):
        fin = open(latentVarFilePath, 'r');
        self.latentQueryDomain = eval(fin.readline().strip());
        self.latentPerceivedScores = eval(fin.readline().strip());
        fin.close();
        return;
        
    def dumpPairPrefLst(self, pairPrefLstPath):
        fout = open(pairPrefLstPath, 'w');
        for (k, l, i1, i2) in self.pairPrefList: fout.write(str((k, l, i1, i2)) + '\n');
        fout.close();
        return;
    
    def loadPairPrefLst(self, pairPrefLstPath, uniWorker=False):
        self.pairPrefList = [];
        fin = open(pairPrefLstPath, 'r');
        for ln in fin:
            if(len(ln.strip()) == 0): continue;
            (k, l, i1, i2) = eval(ln.strip());
            if(not uniWorker): self.pairPrefList.append((k, l, i1, i2));
            else: self.pairPrefList.append((0, l, i1, i2));
        fin.close();
        return self.pairPrefList; 
    
    def dumpModelHumanRead(self, modelFilePath):
        fout = open(modelFilePath, 'w');
        fout.write('paramGoldScores\n');
        for l in self.paramGoldScores:
            fout.write('query: {0}\n'.format(l));
            fout.write(str(self.paramGoldScores[l]) + '\n');
        fout.write('\n');
        fout.write('paramDeltaSquare\n');
        fout.write(str(self.paramDeltaSquare) + '\n');
        fout.write('\n');
        fout.write('paramTau\n');
        for k in self.paramTau:
            fout.write('worker: {0}\n'.format(k));
            fout.write(str(self.paramTau[k]) + '\n');
        fout.write('\n');
        fout.write('paramTheta\n');
        fout.write(str(self.paramTheta) + '\n');
        fout.close();
  
    def rescaleModel(self):  # rescale model parameters, no latent variables
        # rescale
        bias = {};
        for l in self.paramGoldScores:
            gsMin = 1e100;
            gsMax = -1e100;
            for i in self.paramGoldScores[l]:
                if(self.paramGoldScores[l][i] < gsMin): gsMin = self.paramGoldScores[l][i];
                if(self.paramGoldScores[l][i] > gsMax): gsMax = self.paramGoldScores[l][i];
            bias[l] = gsMin;
        scale = math.sqrt(sum(self.paramDeltaSquare.values()) / len(self.paramDeltaSquare));    
        for l in self.paramGoldScores:  # param: gs
            for i in self.paramGoldScores[l]:
                self.paramGoldScores[l][i] = (self.paramGoldScores[l][i] - bias[l]) / scale;
        for l in self.paramDeltaSquare: self.paramDeltaSquare[l] /= (scale ** 2);  # param: delta
        for k in self.paramTau:  # param: tau
            for m in self.paramTau[k]: self.paramTau[k][m] *= scale;
        return;
    


def evaluateKendallTau(gs):
    kt = 0;
    docNum = len(gs);
    for i in range(docNum):
        for j in range(i + 1, docNum):
            if(gs[i] > gs[j]): kt += 1;
    return kt;
        
    
def exp1Gen():
    sim = TPPDataSimulator();
    filePath = os.path.join(os.path.expanduser('~'), 'exp/tpp/exp1/model_gen_q_100.model');
    
    if(not os.path.exists(filePath)):
        print('model gen'); 
        sim.generateModelExp1();
        sim.rescaleModel();
        sim.dumpModel(filePath);
        sim.dumpModelHumanRead(filePath + '.hr');
    sim.loadModel(filePath)
    print('data gen');
    latentQueryDomain = {};
    for l in range(0, 100):
        if(l < 50): latentQueryDomain[l] = 0;
        else: latentQueryDomain[l] = 1;
    sim.generateData(docNum=5, latentQueryDomain=latentQueryDomain, ppqLoad=10);
    sim.dumpPairPrefLst(os.path.join(os.path.expanduser('~'), 'exp/tpp/exp1/model_gen_q_100_29.data'));
    print('finished');
    return;

def exp3Gen():
    sim = TPPDataSimulator();
    docNum = 30;
    dNum = 100;
    wNum = 50;
    sparseRatio = 0.02;
    demographicPmf = [0.2, 0.6, 0.1, 0.1];
    
#     filePath = os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_ori_sparse_0.05_spammer.model');

    filePath = os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_{0}anno_{1}domain.model'.format(wNum, dNum));

    if(not os.path.exists(filePath)):
        print('model gen'); 
        sim.generateModelExp3(docNum=docNum, demographicPmf=demographicPmf, wNum=wNum, dNum=dNum);
        sim.rescaleModel();
        sim.dumpModel(filePath);
        sim.dumpModelHumanRead(filePath + '.hr');
    sim.loadModel(filePath);

    print('data gen');
    sim.generateData(docNum=docNum, ppqLoad=int(docNum * (docNum - 1.0) / 2.0 * sparseRatio));
    for sec in range(1, 6):
        sim.dumpPairPrefLst(os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_{0}anno_{1}domain_sparse_{2}_{3}.data'.format(wNum, dNum, sparseRatio, sec)));
    print('finished');
    return;

    
if __name__ == '__main__':
    exp3Gen();
    pass;
    
    
    
    sim = TPPDataSimulator();
#     sim.generateModel();
# #     sim.dumpModel('/home/xwang95/exp/tpp/sim_model_w10_q_100_d_2_doc_20.model');
# 
#     sim.loadModel(os.path.join(os.path.expanduser('~'), 'exp/tpp/sim_model_w10_q_100_d_2_doc_20.model'));
#     sim.generateData(20);
#     
#     sim.dumpPairPrefLst(os.path.join(os.path.expanduser('~'), 'exp/tpp/sim_model_w10_q_100_d_2_doc_20_ppq_5.data'));
#     sim.dumpLatentVar(os.path.join(os.path.expanduser('~'), 'exp/tpp/sim_model_w10_q_100_d_2_doc_20_ppq_5.latent'));
#     
