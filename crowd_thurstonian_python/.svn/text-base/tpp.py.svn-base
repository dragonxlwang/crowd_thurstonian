'''
Created on Jan 13, 2014

@author: xwang95
'''
import math;
import random;
from simulator import TPPDataSimulator
from toolkit.num.arithmetic import ifInvalidNum
from toolkit.num.probability import multinomialSampling
from toolkit.num.probability import logNormPdf
from toolkit.num.probability import logNormQfunc
from toolkit.num.probability import normQfunc
from toolkit.num.probability import normPdfQfuncRatio
from toolkit.num.rtnorm import rtstdnorm
import sys;
import os;
from crowd_thurstonian.simulator import evaluateKendallTau
import crowd_thurstonian.mqagg;
from toolkit.num.arithmetic import var, std, avg

class TPP(object):
    '''
    Thurstonian Pairwise Preference 
    '''
    # constant
    constWorkerNum = 0;  # not used
    constQueryNum = 0;  # not used
    constDomainNum = 0;
    
    # data: D
    obsSetIdxWorker = {};  # k -> k,l,i1,i2;
    obsSetIdxQuery = {};  # l -> k,l,i1,i2;
    obsSetIdxWorkerQueryWinner = {};  # k => l => i1 -> k,l,i1,i2
    obsSetIdxWorkerQueryLoser = {};  # k => l => i2 -> k,l,i1,i2
    
    # model parameters: Theta
    paramGoldScores = {};  # gold standard scores, index: query, doc; l => i -> score
    paramDeltaSquare = {};  # query difficulty, index: query; l -> delta
    paramTau = {};  # worker param, index: worker; k => m -> tau
    paramTheta = [];  # domain prior, index: query; m -> theta
    
    # latent variables: Z
    latentQueryDomain = {};  # query domain, index: query -> domain; l => m
    latentPerceivedScores = {};  # perceived scores, index: worker, query, doc -> ps; k => l => i -> score
    
    # auxiliary variables: V
    auxNoisyScoreDiff = {};  # noisy scores difference, index: worker, domain query, doc, i1, i2 -> nsDiff; k => l => i1 => i2 -> score diff    
    
    # running configuration
    configBurninIterNum = 100;
    configSampleIterNum = 100;
    configElboEvalInt = 10;
    configDumpModelInt = 10;
    configDumpModelFilePathPatt = "/home/xwang95/exp/tpp/cache_simulate_data_{0}.model";
    # full model v.s. partial model
    configNoOptimizeDelta = False;
    
    
    def __init__(self, wNum, qNum, dNum, pairPrefList,
                 initModelFilePath=None,
                 configBurninIterNum=100,
                 configSampleIterNum=100,
                 configNoOptimizeDelta=False,
                 configElboEvalInt=10,
                 configDumpModelInt=10,
                 configDumpModelFilePathPatt="/home/xwang95/exp/tpp/cache_simulate_data_{0}.model"):
        '''
        Constructor
        '''
        # constant
        self.constWorkerNum = wNum;
        self.constQueryNum = qNum;
        self.constDomainNum = dNum;
        
        # data
        self.obsSetIdxWorker = {};
        self.obsSetIdxQuery = {};
        self.obsSetIdxWorkerQueryWinner = {};
        self.obsSetIdxWorkerQueryLoser = {};
        
        for (k, l, i1, i2) in pairPrefList:
            if(k not in self.obsSetIdxWorker): self.obsSetIdxWorker[k] = [];
            if(l not in self.obsSetIdxQuery): self.obsSetIdxQuery[l] = [];
            if(k not in self.obsSetIdxWorkerQueryWinner): self.obsSetIdxWorkerQueryWinner[k] = {};
            if(l not in self.obsSetIdxWorkerQueryWinner[k]): self.obsSetIdxWorkerQueryWinner[k][l] = {};
            if(i1 not in self.obsSetIdxWorkerQueryWinner[k][l]): self.obsSetIdxWorkerQueryWinner[k][l][i1] = [];
            if(i2 not in self.obsSetIdxWorkerQueryWinner[k][l]): self.obsSetIdxWorkerQueryWinner[k][l][i2] = [];
            if(k not in self.obsSetIdxWorkerQueryLoser): self.obsSetIdxWorkerQueryLoser[k] = {};
            if(l not in self.obsSetIdxWorkerQueryLoser[k]): self.obsSetIdxWorkerQueryLoser[k][l] = {};
            if(i1 not in self.obsSetIdxWorkerQueryLoser[k][l]): self.obsSetIdxWorkerQueryLoser[k][l][i1] = [];
            if(i2 not in self.obsSetIdxWorkerQueryLoser[k][l]): self.obsSetIdxWorkerQueryLoser[k][l][i2] = [];
            self.obsSetIdxWorker[k].append((k, l, i1, i2));
            self.obsSetIdxQuery[l].append((k, l, i1, i2));
            self.obsSetIdxWorkerQueryWinner[k][l][i1].append((k, l, i1, i2));
            self.obsSetIdxWorkerQueryLoser[k][l][i2].append((k, l, i1, i2));
        
        # model parameters
        self.paramGoldScores = {};
        self.paramDeltaSquare = {};
        self.paramTau = {};
        self.paramTheta = {};
        
        # latent variables
        self.latentQueryDomain = {};
        self.latentPerceivedScores = {};
        
        # auxiliary variables
        self.auxNoisyScoreDiff = {};
        
        # running configuration
        self.configBurninIterNum = configBurninIterNum;
        self.configSampleIterNum = configSampleIterNum;
        self.configElboEvalInt = configElboEvalInt;
        self.configDumpModelInt = configDumpModelInt;
        self.configDumpModelFilePathPatt = configDumpModelFilePathPatt;
        self.configNoOptimizeDelta = configNoOptimizeDelta;
        
        # initial model
        if(initModelFilePath is not None): 
            self.loadModel(initModelFilePath);
        else:
            for (k, l, i1, i2) in pairPrefList:
                if(l not in self.paramGoldScores): self.paramGoldScores[l] = {};  # gs
                if(i1 not in self.paramGoldScores[l]): self.paramGoldScores[l][i1] = random.random() + 0.5;
                if(i2 not in self.paramGoldScores[l]): self.paramGoldScores[l][i2] = random.random() + 0.5;
                if(self.configNoOptimizeDelta):
                    if(l not in self.paramDeltaSquare): self.paramDeltaSquare[l] = 1;
                else:
                    if(l not in self.paramDeltaSquare): self.paramDeltaSquare[l] = random.random() + 0.5;  # delta square
                #===============================================================
                # debug
                #===============================================================
                if(k not in self.paramTau): 
                    self.paramTau[k] = {};  # tau
                    for m in range(self.constDomainNum): self.paramTau[k][m] = 1.0;
                    s = sum(self.paramTau[k].values());
                    for m in range(self.constDomainNum): self.paramTau[k][m] /= s;
            self.paramTheta = [1.0 for m in range(self.constDomainNum)];  # theta
            s = sum(self.paramTheta);
            for m in range(self.constDomainNum): self.paramTheta[m] /= s;
        for (k, l, i1, i2) in pairPrefList:
            if(l not in self.latentQueryDomain): self.latentQueryDomain[l] = multinomialSampling(self.paramTheta);  # qd            
            if(k not in self.latentPerceivedScores): self.latentPerceivedScores[k] = {};  # ps
            if(l not in self.latentPerceivedScores[k]): self.latentPerceivedScores[k][l] = {};
            if(i1 not in self.latentPerceivedScores[k][l]): self.latentPerceivedScores[k][l][i1] = self.paramGoldScores[l][i1];
            if(i2 not in self.latentPerceivedScores[k][l]): self.latentPerceivedScores[k][l][i2] = self.paramGoldScores[l][i2];
            if(k not in self.auxNoisyScoreDiff): self.auxNoisyScoreDiff[k] = {};  # nsdiff
            if(l not in self.auxNoisyScoreDiff[k]): self.auxNoisyScoreDiff[k][l] = {};
            if(i1 not in self.auxNoisyScoreDiff[k][l]): self.auxNoisyScoreDiff[k][l][i1] = {};
            self.auxNoisyScoreDiff[k][l][i1][i2] = max(self.paramGoldScores[l][i1] - self.paramGoldScores[l][i2], random.random());  # enforce positivity
        # rescaling
        self.rescaleModel();
        return;
    
    def dumpModel(self, modelFilePath):
        fout = open(modelFilePath, 'w');
        fout.write(str(self.paramGoldScores) + '\n');
        fout.write(str(self.paramDeltaSquare) + '\n');
        fout.write(str(self.paramTau) + '\n');
        fout.write(str(self.paramTheta) + '\n');
        fout.close();
        return;
    
    def loadModel(self, modelFilePath):
        fin = open(modelFilePath, 'r');
        self.paramGoldScores = eval(fin.readline().strip());
        self.paramDeltaSquare = eval(fin.readline().strip());
        self.paramTau = eval(fin.readline().strip());
        self.paramTheta = eval(fin.readline().strip());
        fin.close();
        return;
        
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
    
    def __sgn(self, x): return 1.0 if x >= 0.0 else -1.0;    
    
    def __mMultiPost(self, ll):
        vec = [math.log(self.paramTheta[m]) for m in range(self.constDomainNum)];  # prior
        for (k, l, i1, i2) in self.obsSetIdxQuery[ll]:  # nsdiff generation
            psDiff = self.latentPerceivedScores[k][l][i1] - self.latentPerceivedScores[k][l][i2];
            nsDiff = self.auxNoisyScoreDiff[k][l][i1][i2]; 
            for m in range(len(vec)):
                sgn = self.__sgn(self.paramTau[k][m]);
                var = 2 * (self.paramTau[k][m] ** (-2));
                vec[m] += logNormPdf(nsDiff, sgn * (psDiff), var);        
        shft = max(vec);  # numeric issue: rescaling
        for m in range(self.constDomainNum): vec[m] = math.exp(vec[m] - shft);
        nt = sum(vec);  # normalization
        pmf = [vec[m] / nt for m in range(self.constDomainNum)];
        m = multinomialSampling(pmf);
        return m;        
    
    def __psNormalPost(self, kk, ll, ii):
        mm = self.latentQueryDomain[ll];
        tt = self.paramTau[kk][mm];
        sgn = self.__sgn(tt);
        ww = (tt ** 2) / 2.0;
        # moment
        a1 = self.paramGoldScores[ll][ii] / self.paramDeltaSquare[ll];  # prior
        a2 = 1.0 / self.paramDeltaSquare[ll];
        for (k, l, i1, i2) in self.obsSetIdxWorkerQueryWinner[kk][ll][ii]:  # winner nsdiff generation
            a1 += ww * (self.latentPerceivedScores[k][l][i2] + sgn * self.auxNoisyScoreDiff[k][l][i1][i2]);
            a2 += ww;
        for (k, l, i1, i2) in self.obsSetIdxWorkerQueryLoser[kk][ll][ii]:  # loser nsdiff generation
            a1 += ww * (self.latentPerceivedScores[k][l][i1] - sgn * self.auxNoisyScoreDiff[k][l][i1][i2]);
            a2 += ww;
        # mean, var
        mean = a1 / a2;
        var = 1.0 / a2;
        x = random.gauss(mean, math.sqrt(var));
        return x;
    
    def __nsDiffTruncNormalPost(self, kk, ll, ii1, ii2):
        mm = self.latentQueryDomain[ll];
        tt = self.paramTau[kk][mm];
        sgn = self.__sgn(tt);
        # truncate normal mean and std
        mean = sgn * (self.latentPerceivedScores[kk][ll][ii1] - self.latentPerceivedScores[kk][ll][ii2]); 
        var = 2.0 / (tt ** 2);
        std = math.sqrt(var);
        return mean + std * rtstdnorm(-mean / std, 1e100);  # truncate normal distribution, Mazet's version
    
    def sampleLatentQueryDomain(self):
        # sampling query domain
        for l in self.latentQueryDomain: 
            self.latentQueryDomain[l] = self.__mMultiPost(l);
        return;
    
    def sampleLatentPerceivedScores(self):
        # sampling perceived score
        for k in self.latentPerceivedScores:
            for l in self.latentPerceivedScores[k]:
                for i in self.latentPerceivedScores[k][l]:
                    self.latentPerceivedScores[k][l][i] = self.__psNormalPost(k, l, i);
        return;
    
    def sampleAuxNoisyScoreDiff(self):
        # sample noisy score diff
        for k in self.auxNoisyScoreDiff:
            for l in self.auxNoisyScoreDiff[k]:
                for i1 in self.auxNoisyScoreDiff[k][l]:
                    for i2 in self.auxNoisyScoreDiff[k][l][i1]:
                        self.auxNoisyScoreDiff[k][l][i1][i2] = self.__nsDiffTruncNormalPost(k, l, i1, i2);            
        return;

    def bookkeepingLatentVars(self):
        bkqd = {};
        bkps = {};
        for l in self.latentQueryDomain: bkqd[l] = self.latentQueryDomain[l]; 
        for k in self.latentPerceivedScores:
            if k not in bkps: bkps[k] = {};
            for l in self.latentPerceivedScores[k]:
                if l not in bkps[k]: bkps[k][l] = {};
                for i in self.latentPerceivedScores[k][l]:
                    bkps[k][l][i] = self.latentPerceivedScores[k][l][i];
        return (bkqd, bkps);
        
    def sampleIter(self):
        # Z
        self.sampleLatentQueryDomain();
        self.sampleLatentPerceivedScores();
        # V
        self.sampleAuxNoisyScoreDiff();
        return;
        
    def __gsExact(self, ll, ii, bkqdLst, bkpsLst, tNum):
        s = 0.0;
        c = 0.0;
        for k in self.latentPerceivedScores:
            if(ll not in self.latentPerceivedScores[k]): continue;
            if(ii not in self.latentPerceivedScores[k][ll]): continue;
            for t in range(tNum): s += 1.0 / tNum * bkpsLst[t][k][ll][ii];
            c += 1.0;
        return (s / c);
    
    def __dsExact(self, ll, bkqdLst, bkpsLst, tNum):
        s = 0.0;
        c = 0.0;
        for k in self.latentPerceivedScores:
            if(ll  not in self.latentPerceivedScores[k]): continue;
            for i in self.latentPerceivedScores[k][ll]:
                for t in range(tNum): s += 1.0 / tNum * ((self.paramGoldScores[ll][i] - bkpsLst[t][k][ll][i]) ** 2);
                c += 1.0;
        return (s / c);
    
    def __tauApproxStep(self, kk, mm, tt, bkqdLst, bkpsLst, tNum):
        g1 = 0.0;
        g2 = 0.0;
        for (k, l, i1, i2) in self.obsSetIdxWorker[kk]:
            for t in range(tNum):
                if(bkqdLst[t][l] != mm): continue;
                xx = -(bkpsLst[t][k][l][i1] - bkpsLst[t][k][l][i2]) / math.sqrt(2);
                pq = normPdfQfuncRatio(tt * xx);
                g1 += 1.0 / tNum * (-xx) * pq;
                g2 += 1.0 / tNum * (xx ** 2) * (tt * xx * pq - (pq ** 2));
        if(g1 == 0.0 and g2 == 0.0): return 0.0;  # no data supported
        #=======================================================================
        # TODO: possibly error here: g1 != 0.0 but g2 == 0.0
        #=======================================================================
        return  (-g1 / g2);
    
    def __tauApproxVecStep(self, kk, tt, bkqdLst, bkpsLst, tNum):
        g1 = [0.0 for m in range(self.constDomainNum)];
        g2 = [0.0 for m in range(self.constDomainNum)];
        ss = [0.0 for m in range(self.constDomainNum)];
        for (k, l, i1, i2) in self.obsSetIdxWorker[kk]:
            for t in range(tNum):
                mm = bkqdLst[t][l];
                xx = -(bkpsLst[t][k][l][i1] - bkpsLst[t][k][l][i2]) / math.sqrt(2);
                pq = normPdfQfuncRatio(tt[mm] * xx);
                g1[mm] += 1.0 / tNum * (-xx) * pq;
                g2[mm] += 1.0 / tNum * (xx ** 2) * (tt[mm] * xx * pq - (pq ** 2));
                if(ifInvalidNum(xx) or ifInvalidNum(pq)):
                    print(xx, pq, g1[mm], g2[mm], k, l, i1, i2, tt[mm], mm);
                    sys.stdin.read();
        for m in range(self.constDomainNum):
            if(g1[m] == 0.0 and g2[m] == 0.0): ss[m] = 0.0;  # no data supported
            else: ss[m] = (-g1[m] / g2[m]);
            #=======================================================================
            # TODO: possibly error here: g1 != 0.0 but g2 == 0.0
            #=======================================================================
        return  ss;
    
    def __tauApproxNew(self, kk, mm, tt, bkqdLst, bkpsLst, tNum):
        eps = 1e-2;
        while(True):
            ss = self.__tauApproxStep(kk, mm, tt, bkqdLst, bkpsLst, tNum);
            tt = tt + ss;
            if(abs(ss) <= eps): return tt;
        return;
    
    def __tauApproxVecNew(self, kk, tt, bkqdLst, bkpsLst, tNum):
        eps = 1e-2;
        iter = 0;
        maxSLst = [];
        while(True):
            iter += 1;
            ss = self.__tauApproxVecStep(kk, tt, bkqdLst, bkpsLst, tNum);
            maxS = max([abs(x) for x in ss]);
            maxSLst.append(maxS);
            #===================================================================
            # handle special cases 1
            #===================================================================
            if(maxS > 1e10):
                print('tau approx vec: large step error'); 
                return tt;
            #===================================================================
            # 
            #===================================================================
            for m in range(self.constDomainNum): tt[m] = tt[m] + ss[m];
            #===================================================================
            # handle special cases 2
            #===================================================================
            if(iter > 10):
                print('tau approx vec: infinity step error: {0}'.format(maxSLst));
                return tt;
            #===================================================================
            # 
            #===================================================================
            if(maxS <= eps): return tt;
        return;
    
    def __thetaExact(self, mm, bkqdLst, bkpsLst, tNum):
        s = 0.0;
        for l in self.latentQueryDomain:
            for t in range(tNum): s += ((1.0 / tNum) if(bkqdLst[t][l] == mm) else 0.0);
        return s;
    
    def __thetaExactVec(self, bkqdLst, bkpsLst, tNum):
        eps = 1e-6;
        ss = [eps for m in range(self.constDomainNum)];
        for l in self.latentQueryDomain:
            for t in range(tNum): ss[bkqdLst[t][l]] += 1.0;
        s = sum(ss);
        return [ss[m] / s for m in range(self.constDomainNum)];
        
    def updateParamGoldScores(self, bkqdLst, bkpsLst, tNum):
        for l in self.paramGoldScores:
            for i in self.paramGoldScores[l]:
                self.paramGoldScores[l][i] = self.__gsExact(l, i, bkqdLst, bkpsLst, tNum);
        return;
    
    def updateParamDeltaSquare(self, bkqdLst, bkpsLst, tNum):
        for l in self.paramDeltaSquare: self.paramDeltaSquare[l] = self.__dsExact(l, bkqdLst, bkpsLst, tNum);
        return;
    
    def updateParamTau(self, bkqdLst, bkpsLst, tNum, ifVecCompute=True):
        if(ifVecCompute):
            # vec version
            for k in self.paramTau:
                sys.stdout.write('*');
                sys.stdout.flush();
                tt = self.__tauApproxVecNew(k, self.paramTau[k], bkqdLst, bkpsLst, tNum);
                for m in range(self.constDomainNum): self.paramTau[k][m] = tt[m];
        else:
            # scalar
            for k in self.paramTau:
                for m in self.paramTau[k]:
                    sys.stdout.write('*');
                    sys.stdout.flush();
                    self.paramTau[k][m] = self.__tauApproxNew(k, m, self.paramTau[k][m], bkqdLst, bkpsLst, tNum);
        print('');
        return;
    
    def updateParamTheta(self, bkqdLst, bkpsLst, tNum, ifVecCompute=True):
        if(ifVecCompute):
            vec = self.__thetaExactVec(bkqdLst, bkpsLst, tNum);
            for m in range(self.constDomainNum): self.paramTheta[m] = vec[m];
        else:
            eps = 1e-6;
            vec = [self.__thetaExact(m, bkqdLst, bkpsLst, tNum) for m in range(self.constDomainNum)];
            s = sum(vec);
            for m in range(self.constDomainNum): self.paramTheta[m] = (1 - eps) * vec[m] / s + eps / self.constDomainNum;
        return;
      
    IF_FOLD = False;
    #===========================================================================
    # E-M
    #===========================================================================
    def eStep(self, burninIterNum=100, sampleIterNum=100):
        bkqdLst = [];
        bkpsLst = [];
        # burn-in
        for t in range(burninIterNum):
            self.sampleIter();
            sys.stdout.write('*');
            sys.stdout.flush();
        sys.stdout.write('^');
        sys.stdout.flush();
        # sampling
        for t in range(sampleIterNum):
            self.sampleIter();
            (bkqd, bkps) = self.bookkeepingLatentVars();
            bkqdLst.append(bkqd);
            bkpsLst.append(bkps);
            sys.stdout.write('*');
            sys.stdout.flush();
        print('');
        return (bkqdLst, bkpsLst);
    
    def mStep(self, bkqdLst, bkpsLst, tNum):
#         print('paramGoldScores') 
        self.updateParamGoldScores(bkqdLst, bkpsLst, tNum);
#         print('paramDeltaSquare')
        if(not self.configNoOptimizeDelta):  # ignore question difficulty
            self.updateParamDeltaSquare(bkqdLst, bkpsLst, tNum);
#         print('paramTau')
        self.updateParamTau(bkqdLst, bkpsLst, tNum);
#         print('paramTheta')
        self.updateParamTheta(bkqdLst, bkpsLst, tNum);
        return;
    
    def rescaleModel(self):
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
        for k in self.latentPerceivedScores:  # latent: ps
            for l in self.latentPerceivedScores[k]:
                for i in self.latentPerceivedScores[k][l]:
                    self.latentPerceivedScores[k][l][i] = (self.latentPerceivedScores[k][l][i] - bias[l]) / scale;
        return;
            
    def evalElbo(self, bkqdLst, bkpsLst, tNum):
        elbo = 0.0;
        for l in self.latentQueryDomain: 
            for t in range(tNum): elbo += 1.0 / tNum * math.log(self.paramTheta[bkqdLst[t][l]]);
        for k in self.latentPerceivedScores:
            for l in self.latentPerceivedScores[k]:
                for i in self.latentPerceivedScores[k][l]:
                    mean = self.paramGoldScores[l][i];
                    var = self.paramDeltaSquare[l];
                    for t in range(tNum): elbo += 1.0 / tNum * logNormPdf(bkpsLst[t][k][l][i], mean, var);
        for l in self.obsSetIdxQuery:
            for (k, l, i1, i2) in self.obsSetIdxQuery[l]:
                for t in range(tNum):
                    xx = -(bkpsLst[t][k][l][i1] - bkpsLst[t][k][l][i2]) / math.sqrt(2);
                    mm = bkqdLst[t][l];
                    tt = self.paramTau[k][mm];
                    elbo += 1.0 / tNum * logNormQfunc(tt * xx);
        return elbo;
    
    def iter(self, ifEval=False):
        elbo = None;
        burninIterNum = self.configBurninIterNum;
        sampleIterNum = self.configSampleIterNum;
        sys.stdout.write('E: ');
        sys.stdout.flush();
        (bkqdLst, bkpsLst) = self.eStep(burninIterNum, sampleIterNum);  # E-step
        sys.stdout.write('M: ');
        sys.stdout.flush();
        self.mStep(bkqdLst, bkpsLst, sampleIterNum);  # M-step
        if(ifEval):
            sys.stdout.write('ELBO: ');
            sys.stdout.flush();
            elbo = self.evalElbo(bkqdLst, bkpsLst, sampleIterNum);
            print(elbo);
        print('RESCALE');
        self.rescaleModel();  # rescale
        return elbo;
    
    def infer(self):
        t = 0;
        while(True):
            print('||=> Iter = {0}'.format(t));
            if(t % self.configElboEvalInt == 0): self.iter(True);
            else: self.iter(False);
            #===================================================================
            # evaluation code:
            #===================================================================
            evalParamGoldScores(self.paramGoldScores);
            evalWorkerExpertise(self.paramTau);
            #===================================================================
            # 
            #===================================================================
            if(t % self.configDumpModelInt == 0):
                modelFilePath = self.configDumpModelFilePathPatt.format(t);
                print('DUMP AT: {0}'.format(modelFilePath));
                self.dumpModel(modelFilePath);
                self.dumpModelHumanRead(modelFilePath + '.hr')
            t += 1;
            print('');
        return;
    
    __debug = False;
    
    
#===============================================================================
# 
#===============================================================================
def exp1Infer(sec=4, configBurninIterNum=50, configSampleIterNum=50):
    print('sec={0}'.format(sec));
    pairPrefLstPath = os.path.join(os.path.expanduser('~'), 'exp/tpp/exp1/model_gen_q_100_{0}.data'.format(sec));
    sim = TPPDataSimulator();
    sim.loadPairPrefLst(pairPrefLstPath);
    tpp = TPP(wNum=5,
              qNum=100,
              dNum=2,
              pairPrefList=sim.pairPrefList,
              configBurninIterNum=configBurninIterNum,
              configSampleIterNum=configSampleIterNum,
              configElboEvalInt=1,
              configDumpModelInt=1,
              configDumpModelFilePathPatt=os.path.join(os.path.expanduser('~'),
                                                       'exp/tpp/exp1/model_gen_q_100_' 
                                                       + str(sec) 
                                                       + '_burnin_' + str(configBurninIterNum)
                                                       + '_sample_' + str(configSampleIterNum) 
                                                       + '_iter_{0}.model.estimate'),
              initModelFilePath=None);
    tpp.infer();
    return;

def exp3Infer(sec=1, configBurninIterNum=50, configSampleIterNum=50):
    print('50anno_sparse_0.02');
    print('sec={0}'.format(sec));
    #===========================================================================
    # full model setting
    #===========================================================================
    wNum = 50;  # if ignore individual worker
    dNum = 5;  # if ignore domain
    configNoOptimizeDelta = False;  # if ignore query difficulty
    #===========================================================================
    # parital model baselines
    #===========================================================================
#     dNum = 1; # no domain
#     wNum = 1;  # one worker
#     configNoOptimizeDelta = True;  # unidifficult query
    #===========================================================================
    # 
    #===========================================================================
    pairPrefLstPath = os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_50anno_sparse_0.02_{0}.data'.format(sec));
    
    configDumpModelFilePathPatt = os.path.join(os.path.expanduser('~'),
                                                       'exp/tpp/exp3/model_50anno_sparse_0.02_'
                                                       + str(sec) 
                                                       + '_burnin_' + str(configBurninIterNum)
                                                       + '_sample_' + str(configSampleIterNum) 
                                                       + '_iter_{0}.model.estimate');    
    #===========================================================================
    # rw-rw----. 1 xwang95 xwang95 5300920 Jan 17 22:17 model_gen_ori_1.data
    # -rw-rw----. 1 xwang95 xwang95  320111 Jan 18 01:02 model_gen_ori_sparse_0.05_1.data
    # -rw-rw----. 1 xwang95 xwang95  319921 Jan 18 03:23 model_gen_ori_sparse_0.05_malicious_1.data
    # -rw-rw----. 1 xwang95 xwang95  319938 Jan 18 03:24 model_gen_ori_sparse_0.05_spammer_1.data
    # -rw-rw----. 1 xwang95 xwang95  655289 Jan 18 01:13 model_gen_ori_sparse_0.1_1.data
    # -rw-rw----. 1 xwang95 xwang95  655289 Jan 18 01:13 model_gen_ori_sparse_0.1_2.data
    # -rw-rw----. 1 xwang95 xwang95  655289 Jan 18 01:13 model_gen_ori_sparse_0.1_3.data
    # -rw-rw----. 1 xwang95 xwang95  655289 Jan 18 01:13 model_gen_ori_sparse_0.1_4.data
    # -rw-rw----. 1 xwang95 xwang95  655289 Jan 18 01:13 model_gen_ori_sparse_0.1_5.data
    # -rw-rw----. 1 xwang95 xwang95  111200 Jan 17 23:22 model_gen_short_10.data
    # -rw-rw----. 1 xwang95 xwang95  111200 Jan 17 23:21 model_gen_short_1.data
    # -rw-rw----. 1 xwang95 xwang95   69500 Jan 17 23:54 model_gen_short_sparse_0.5_1.data
    # -rw-rw----. 1 xwang95 xwang95   69500 Jan 18 03:22 model_gen_short_sparse_malicious_1.data
    # -rw-rw----. 1 xwang95 xwang95   69500 Jan 18 03:19 model_gen_short_sparse_spammer_1.data
    #===========================================================================
    print('data file: {0}'.format(pairPrefLstPath));
    print('model file: {0}'.format(os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_50anno.model')));
    sim = TPPDataSimulator();
    sim.loadPairPrefLst(pairPrefLstPath, True if(wNum == 1) else False);
    tpp = TPP(wNum=wNum,
              qNum=100,
              dNum=dNum,
              pairPrefList=sim.pairPrefList,
              configBurninIterNum=configBurninIterNum,
              configSampleIterNum=configSampleIterNum,
              configNoOptimizeDelta=configNoOptimizeDelta,
              configElboEvalInt=1,
              configDumpModelInt=1,
              configDumpModelFilePathPatt=configDumpModelFilePathPatt,
              initModelFilePath=None);
    tpp.infer();
    return;

def expMQ2008Agg(sec=1, configBurninIterNum=50, configSampleIterNum=50):
    print('mq2008agg_small_sparse_0.2_domain_5');
    print('sec={0}'.format(sec));
    #===========================================================================
    # full model setting
    #===========================================================================
    wNum = 25;  # if ignore individual worker
    dNum = 5;  # if ignore domain
    configNoOptimizeDelta = False;  # if ignore query difficulty
    print('dNum={0}'.format(dNum));
    #===========================================================================
    # parital model baselines
    #===========================================================================
#     dNum = 1; # no domain
#     wNum = 1;  # one worker
#     configNoOptimizeDelta = True;  # unidifficult query
    #===========================================================================
    # 
    #===========================================================================
    pairPrefLstPath = os.path.join(os.path.expanduser('~'), 'exp/tpp/mq2008agg_small',
                                   'mq2008agg_small_sparse_0.2_{0}.data'.format(sec));
    
    configDumpModelFilePathPatt = os.path.join(os.path.expanduser('~'), 'exp/tpp/mq2008agg_small',
                                               'mq2008agg_small_sparse_0.2_domain_5_'
                                               + str(sec)
                                               + '_burnin_' + str(configBurninIterNum)
                                               + '_sample_' + str(configSampleIterNum)
                                               + '_iter_{0}.model.estimate');
        

    print('data file: {0}'.format(pairPrefLstPath));
    
    pairPrefLst = crowd_thurstonian.mqagg.loadPairPrefLst(pairPrefLstPath, True if(wNum == 1) else False);
    tpp = TPP(wNum=wNum,
              qNum=403,
              dNum=dNum,
              pairPrefList=pairPrefLst,
              configBurninIterNum=configBurninIterNum,
              configSampleIterNum=configSampleIterNum,
              configNoOptimizeDelta=configNoOptimizeDelta,
              configElboEvalInt=1,
              configDumpModelInt=1,
              configDumpModelFilePathPatt=configDumpModelFilePathPatt,
              initModelFilePath=None);
    tpp.infer();
    return;

def evalWorkerExpertise(paramTau):
    sim = TPPDataSimulator();
    sim.loadModel(os.path.join(os.path.expanduser('~'), 'exp/tpp/exp3/model_gen_50anno.model'));
    gtMaliciousLabMalicious = 0.0;
    gtMaliciousLabTruth = 0.0;
    gtTruthLabMalicious = 0.0;
    gtTruthLabTruth = 0.0;
    # confusion table
    for k in paramTau:
        for m in paramTau[k]:
            if(sim.paramTau[k][m] > 0.0 and paramTau[k][m] > 0.0): gtTruthLabTruth += 1.0;
            if(sim.paramTau[k][m] > 0.0 and paramTau[k][m] < 0.0): gtTruthLabMalicious += 1.0;
            if(sim.paramTau[k][m] < 0.0 and paramTau[k][m] > 0.0): gtMaliciousLabTruth += 1.0
            if(sim.paramTau[k][m] < 0.0 and paramTau[k][m] < 0.0): gtMaliciousLabMalicious += 1.0;
    # recall and precision
    if(gtTruthLabMalicious + gtMaliciousLabMalicious != 0.0): precMalicious = gtMaliciousLabMalicious / (gtTruthLabMalicious + gtMaliciousLabMalicious);
    else:  precMalicious = 0.0;
    if(gtMaliciousLabMalicious + gtMaliciousLabTruth != 0.0): recallMalicious = gtMaliciousLabMalicious / (gtMaliciousLabMalicious + gtMaliciousLabTruth);
    else: recallMalicious = 0.0;
    if(precMalicious + recallMalicious == 0.0): f1Malicious = 0.0;
    else: f1Malicious = 2.0 * precMalicious * recallMalicious / (precMalicious + recallMalicious);
    
    if(gtTruthLabTruth + gtMaliciousLabTruth != 0.0): precTruth = gtTruthLabTruth / (gtTruthLabTruth + gtMaliciousLabTruth);
    else: precTruth = 0.0;
    if(gtTruthLabTruth + gtTruthLabMalicious != 0.0): recallTruth = gtTruthLabTruth / (gtTruthLabTruth + gtTruthLabMalicious);
    else: recallTruth = 0.0;
    if(precTruth + recallTruth == 0.0) : f1Truth = 0.0;
    else: f1Truth = 2.0 * precTruth * recallTruth / (precTruth + recallTruth);
    
    acc = (gtTruthLabTruth + gtMaliciousLabMalicious) / (gtTruthLabTruth + gtMaliciousLabMalicious + gtTruthLabMalicious + gtMaliciousLabTruth);
    
    print('MALICIOUS: prec = {0}; recall = {1}; f1 = {2}'.format(precMalicious, recallMalicious, f1Malicious));
    print('TRUTH: prec = {0}; recall = {1}; f1 = {2}'.format(precTruth, recallTruth, f1Truth));
    print('ACC: acc = {0}'.format(acc));
    return;
    
    
def evalParamGoldScores(paramGoldScores):
    #===========================================================================
    # for simulated data
    #===========================================================================
    ktLst = [];
    for l in paramGoldScores: ktLst.append(evaluateKendallTau([paramGoldScores[l][i] for i in sorted(paramGoldScores[l])]));
    (avgKt, stdKt) = (avg(ktLst), std(ktLst));
    print('KENDALL TAU: avg = {0}; std = {1}'.format(avgKt, stdKt));

    return (avgKt, stdKt);

    #===========================================================================
    # for MQ2008small
    #===========================================================================
#     paramGS = crowd_thurstonian.mqagg.loadParamGS();
#     ndcg2Lst = [];
#     ndcg4Lst = [];
#     ndcg6Lst = [];
#     ndcg8Lst = [];
#     for l in paramGoldScores: 
#         ndcg2Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(paramGoldScores[l], paramGS[l], 2));
#         ndcg4Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(paramGoldScores[l], paramGS[l], 4));
#         ndcg6Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(paramGoldScores[l], paramGS[l], 6));
#         ndcg8Lst.append(crowd_thurstonian.mqagg.evaluteNDCG(paramGoldScores[l], paramGS[l], 8));
#     ndcg2Lst = [x for x in ndcg2Lst if(x is not None)];
#     ndcg4Lst = [x for x in ndcg4Lst if(x is not None)];
#     ndcg6Lst = [x for x in ndcg6Lst if(x is not None)];
#     ndcg8Lst = [x for x in ndcg8Lst if(x is not None)];
#     (avgNdcg2, stdNdcg2) = (avg(ndcg2Lst), std(ndcg2Lst));
#     (avgNdcg4, stdNdcg4) = (avg(ndcg4Lst), std(ndcg4Lst));
#     (avgNdcg6, stdNdcg6) = (avg(ndcg6Lst), std(ndcg6Lst));
#     (avgNdcg8, stdNdcg8) = (avg(ndcg8Lst), std(ndcg8Lst));
#     print('NDCG@2: avg = {0}; std = {1}'.format(avgNdcg2, stdNdcg2));
#     print('NDCG@4: avg = {0}; std = {1}'.format(avgNdcg4, stdNdcg4));
#     print('NDCG@6: avg = {0}; std = {1}'.format(avgNdcg6, stdNdcg6));
#     print('NDCG@8: avg = {0}; std = {1}'.format(avgNdcg8, stdNdcg8));
    return;
    
if __name__ == '__main__':
#     expMQ2008Agg(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]));
    
    exp3Infer(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]));
    
    
#     exp1Infer(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]));
    
#     if(True):
#         sim = TPPDataSimulator();
#         pairPrefLstPath = os.path.join(os.path.expanduser('~'), 'exp/tpp/sim_model_w10_q_100_d_2_doc_20_ppq_5.data');
#         sim.loadPairPrefLst(pairPrefLstPath);
#         tpp = TPP(wNum=10,
#                   qNum=1000,
#                   dNum=2,
#                   pairPrefList=sim.pairPrefList,
#                   configBurninIterNum=20,
#                   configSampleIterNum=20,
#                   configElboEvalInt=1,
#                   configDumpModelInt=1,
#                   configDumpModelFilePathPatt=os.path.join(os.path.expanduser('~'), "exp/tpp/sim_model_w10_q_100_d_2_doc_20_iter_{0}.model.estimate"),
#                   initModelFilePath=os.path.join(os.path.expanduser('~'), 'exp/tpp/sim_model_w10_q_100_d_2_doc_20_iter_7.model.estimate'));
#         tpp.infer();

        
