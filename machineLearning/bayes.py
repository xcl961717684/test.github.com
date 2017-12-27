# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:43:53 2017
多项式贝叶斯
@author: xcl
"""
import numpy as np
def getFeature(data,feature):
    mydata=[]
    for dtr in data:
        myfeature=[]
        dtr=dtr.split()
        for i in feature:
            myfeature.append(dtr.count(i))
        mydata.append(myfeature)
    return np.array(mydata)
        

def train(data,dataLabel,className):
    classFrequence=[]
    _,t=data.shape
    frequence=np.zeros((len(className),t))
    probability=np.zeros((len(className),t))
    for i in className:
        classFrequence.append(dataLabel.count(i))
    classFrequence=np.array(classFrequence)   
    classProbability=classFrequence/np.sum(classFrequence)
    
    dataLabel=np.array(dataLabel)
    for i,value in enumerate(className):
        index=value==dataLabel
        index=[x for x,y in enumerate(index) if y==True]
        frequence[i]=np.sum(data[index,:],0)
        probability[i]=(frequence[i]+1) /(np.sum(frequence[i])+t)
    return classProbability,probability
    
def test(data,classProbability,probability):
    print("Un-normalized:")
    result=[]
    testCount,t=data.shape
    classCount=classProbability.size
    for te in range(testCount):
        for c in range(classCount):
            p=classProbability[c]
            for i in range(t):
                p=p*pow(probability[c][i],data[te][i])
            result.append(p)
            print("P(c%d|d%d)=%.8f" %(c+1,te+1,p))
    return result

def normalize(result,testCount,classCount):
    print("Normalized:")
    i=0
    testSum=[]
    for te in range(testCount):
        testSum.append(0)
        for c in range(classCount):
            testSum[te]=testSum[te]+result[i]
            i=i+1
    i=0
    #print(testSum)
    for te in range(testCount):
        for c in range(classCount):
            print("P(c%d|d%d)=%.7f" %(c+1,te+1,result[i]/testSum[te]))
            i=i+1
    
def putoutResult(result,testCount,classCount,className):
    print("Result:")
    for start in range(testCount):
        temp=result[start*classCount:(start*classCount+classCount)]
        tempMax=max(temp)
        if tempMax==1/classCount:
            print("test%d is an equal probability event" %(start+1))
            continue
        for x,y in enumerate(temp):
            if tempMax==y:
                print("test%d belong to %s" %(start+1,className[x]))
  

if __name__=='__main__':
    traindata=['Chinese Beijing Chinese','Chinese Chinese Shanghai','Chinese Masco','Tokyo Japan Chinese']
    traindataLabel=['C','C','C','J']
    testdata=['Chinese Chinese Chinese Tokyo Japan','Tokyo Tokyo Japan Shanghai']
    feature=['Beijing','Chinese','Japan','Masco','Shanghai','Tokyo']
    className=['C','J']
    mytraindata=getFeature(traindata,feature)
    mytestdata=getFeature(testdata,feature)
    
    classProbability,probability=train(mytraindata,traindataLabel,className)
    out=np.concatenate((classProbability.reshape(-1,1),probability),axis=1)
    print(out)
    result=test(mytestdata,classProbability,probability)
    normalize(result,len(testdata),len(className))
    putoutResult(result,len(testdata),len(className),className)
    
    
