import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
import station1_ETL as s1
import station2_features as s2
import station3_model as s3
import station3_helpers as s3h
import math


def station4_implementation(clientId, capital):
    '''
    Params: 
    Returns:
    '''
    
    # Get client, equity, indicator and news data
    df1 = s1.station1_ETL()
    
    # Get client, equity, indicator and news feature
    rets = s2.station2_returns(df1['Equity Data'])
    
    clients = df1['Client Data']
    
    opWeights = []
    

    for i, client in enumerate(clients['client_ID']):
        
        if (client == clientId):
        
            clientDetails = clients.loc[i]
            weights = clientDetails[3:12].to_numpy()
            
            # Call the optimisation model function with client risk aversion
            # as a parameter
            print(clientDetails['risk_profile'])
            
            riskyPort = s3.station3_optRiskyPortfolio(rets)
            
            opWeights = s3.station3_optCompletePortfolio(
                riskyPort, 
                clientDetails['risk_profile']
            )
            
    
        # This function will return the optimal weights for the portfolio
        # based on client risk and also output various plots and graphs

    station4_portfolioReturns(rets, opWeights['Risky'], opWeights['Risk-Free'], capital)

    return opWeights   


def station4_portfolioReturns(rets, weights, rfrWeight, cap):
    
    pRets = []
    
    for row in rets.iterrows():
        pRets.append(np.dot(row[1].to_numpy(), weights))
    
    #include rfr in prets
    p = np.array(pRets)
    p = p + (rfrWeight*(0.033/252))    
    
    prices = []
    for r in p:
        prices.append(cap*(1+r))
    
    newP = [x for x in prices if math.isnan(x) == False]
    
    print("----------------- Portfolio Simulation over",len(newP) ,"Days ------------------")
    print("Portfolio Starting Value: ", cap)
    print("Portfolio Lowest Value: ", min(newP))
    print("Portfolio Highest Value: ", max(newP))
    print("Portfolio Ending Value: ", prices[len(newP)-1])
    
    priceSim = pd.DataFrame(newP)
    
    plt.figure(figsize=(25, 10))
    plt.plot(priceSim)
    plt.title("Portfolio Value Simulation")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.show()
    
    
    return priceSim

def getClientWeights(client):
    weights = []
    weights.append(client[3:6])
    return weights


if __name__ == '__main__':
    station4_implementation(clientId=100, capital=150000)
    
    
    
    
    
    