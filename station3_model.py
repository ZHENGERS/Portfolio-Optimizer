import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
import station1_ETL as s1
import station2_features as s2
import station3_helpers as s3helpers
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk



"""
Created on Mon Jul  7 12:02:10 2022

@author: William Zheng (z5313015)
"""

# Calculate the optimal risky portfolio (highest Sharpe ratio)
def station3_optRiskyPortfolio(df):
    """
    Actual Model
    params : 
    return: 
    """
        
    optRiskyPort = {
        'Weights': [],
        'Standard Deviation': 0,
        'Expected Return': 0, 
        'Sharpe Ratio': 0
    }
    
    returns = []
    stds = []
    w = []
    weights = s3helpers.generateWeights(df)

    for i in range(2500):
        weights = s3helpers.generateWeights(df)
        returns.append(s3helpers.portfolioReturn(weights, df))
        stds.append(s3helpers.portfolioStd(weights, df))
        w.append(weights)
    
    returns = np.array(returns)
    stds = np.array(stds)
    print("MIN: ", min(stds))
    # Efficient Frontier
    s3helpers.plotEfficientFrontier(stds, returns)
    
    
    # Make sure that no asset is more than 100% of the total portfolio
    bounds = tuple((0,1) for stock in range(len(df.columns)))
    
    # Define the constraints, ensure weights do not exceed 100%
    constraints = ({'type': 'eq', 'fun': s3helpers.checkSum})
    
    # Define initial guesses
    init_guess = len(df.columns) * [1. / len(df.columns)]
    
    #Perform optimisation          
    optimized_sharpe = sco.minimize(s3helpers.getNegSharpe,init_guess,method='SLSQP',
        bounds = bounds, constraints = constraints)    

    print("***Maximization of Sharpe Ratio***")
    optRiskyPortWeights = optimized_sharpe['x']
    optRiskyPortStats = s3helpers.getPotfolioStats(optRiskyPortWeights)
    optRiskyPort['Weights'] = optRiskyPortWeights
    optRiskyPort['Expected Return'] = optRiskyPortStats[0]
    optRiskyPort['Standard Deviation'] = optRiskyPortStats[1]
    optRiskyPort['Sharpe Ratio'] = optRiskyPortStats[2]
    
    print(f"Optimal Risky Weights: {optRiskyPortWeights.round(3)}")
    print(f"E(r) = {optRiskyPortStats[0]} Std = {optRiskyPortStats[1]} Sharpe = {optRiskyPortStats[2]}")

    
    optimized_var = sco.minimize(s3helpers.getMinVar,init_guess,method = 'SLSQP',
        bounds = bounds,constraints = constraints)  
    
    print("****Minimizing Variance***")
    minVarPortWeights = optimized_var['x'].round(3)
    minVarPortStats = s3helpers.getPotfolioStats(minVarPortWeights)
    print(minVarPortWeights)
    print(f"E(r) = {minVarPortStats[0]} Std = {minVarPortStats[1]} Sharpe = {minVarPortStats[2]}")


    # Return portfolio weights, portfolio return, portfolio std-dev
    print(optRiskyPort)
    return optRiskyPort


# Plot the Capital allocation line and Calculate 
# the optimal complete portfolio (consisting of rf assets and risky)
def station3_optCompletePortfolio(completePort, A):
    optCompletePort = {
        'Risky': [],
        'Risk-Free': 0,
        'Standard Deviation': 0,
        'Expected Return': 0 
    }
    
    # Based of current 10 year Government Bond Yield in Australia
    rfr = 0.03302
    
    # Optimal weight in risky portfolio
    y = (completePort['Expected Return'] - rfr)/(A*completePort['Standard Deviation']**2)
    
    # Optimal weight at risk-free rate
    x = 1-y
    
    optCompletePort['Risk-Free'] = x
    optCompletePort['Expected Return'] = x*rfr + y*completePort['Expected Return']
    
    for w in completePort['Weights']:
        optCompletePort['Risky'].append(w*y)
        
    print("Optimal Weight in risky port: ", y)
    print("Optimal Weight at rfr: ", x)
    #print(optCompletePort)
    
    return optCompletePort



def station3_processNews(df):
    nltk.download('vader_lexicon')
    
    df.columns = ['Equity','Source', 'Date', 'Headline']
    
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = df['Headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    
    print(scores_df.head(20))
    
    # Join the DataFrames of the news and the list of dicts
    df = df.join(scores_df, rsuffix='_right')
    
    fileDB = 'C:/Users/Will/OneDrive/Desktop/FINS3645/Project/Data/'
    df.to_json(fileDB + 'vader_scores.json')
    
    sentArr = []
    
    for sent in df['compound']:
        if (sent > 0):
            sentArr.append("Positive")
        elif (sent < 0):
            sentArr.append("Negative")
        else:
            sentArr.append("Neutral")
    
    df['Sentiment'] = sentArr
    
    df.sort_values(by = ['Date'], ascending = False)
    df.index = df['Date']

    print(df.head(40))
    
    df2 = df
    df2.to_excel("output.xlsx") 

    return df


        
'''    
    cal_x = []
    cal_y = []
    utility = []
    a = 10
    
    print(f"MAX RETURN = {max(returns) }")
    for er in np.linspace(0.025, max(returns), 50):
        sd = (er - 0.025)/optRiskyPortStats[2]
        u = er - .5*a*(sd**2)
        cal_x.append(sd)
        cal_y.append(er)
        utility.append(u)
        
    data2 = {'utility':utility, 'cal_y':cal_y, 'cal_x':cal_x}
    cal = pd.DataFrame(data2)
    print(cal.head())
            
    
    plt.plot(cal_x, cal_y, color='r')

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
    '''
    



# Call other functions
def station3_model():
    
    return 0
    
    


if __name__ == '__main__':    
    features = s2.station2_features(s1.station1_ETL())   
    port = station3_optRiskyPortfolio(features['Returns'])
    station3_optCompletePortfolio(port, 1)
    
    station3_processNews(s1.station1_ETL()['News Data'])
    
    
    
    
    
    