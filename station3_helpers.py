import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
import station1_ETL as s1
import station2_features as s2


"""
Created on Mon Jul  11 11:07:20 2022

@author: William Zheng (z5313015)
"""

def plotEfficientFrontier(stds, returns):
    plt.figure(figsize=(16, 8))
    plt.scatter(stds, returns, c=returns / stds, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe ratio')
    plt.title('Efficient Frontier')
    
    # Minimum Variance portfolio
    plt.scatter(min(stds), returns[np.where(stds == min(stds))], marker='*',
                color='r',s=500)
    plt.show()
    
def getPotfolioStats(weights):
    weights = np.array(weights)
    
    df = getDf()
    ret = portfolioReturn(weights, df)
    std = portfolioStd(weights, df)
    
    sr = (ret - 0.025) / std

    return np.array([ret, std, sr])

def getNegSharpe(weights):
    return -getPotfolioStats(weights)[2]

def getMinVar(weights):
    return getPotfolioStats(weights)[1] ** 2

def getMinFuncPort(weights):
    return getPotfolioStats(weights)[1]

def portfolioReturn(weights, df):
    return np.dot(df.mean(), weights)*252
    

def portfolioStd(weights, df):
    return np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))

    #var = np.dot(np.dot(df.cov(), weights), weights)
    #return var**(1/2) * np.sqrt(252)

def checkSum(weights):
    return np.sum(weights)-1


def getDf():
    fileDB = 'C:/Users/Will/OneDrive/Desktop/FINS3645/Project/Data/'
    df = s2.station2_returns(s1.station1_ETL_equities(fileDB))   
    return df


def generateWeights(df):
    weights = np.random.random(len(df.columns))
    weights /= np.sum(weights)
    return weights





