import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import station1_ETL as s1

"""
Created on Mon Jul  5 18:11:09 2022

@author: William Zheng (z5313015)
"""
    
def station2_returns(df):
    """
    Receive cleaned data from Station #1 process all relevant features
    param: df - input clean data streams
    return: df - return relevant features
    """
    
    rets = pd.DataFrame()
    
    i = 0
    for col in df:
            
        if 'PX_LAST' in col and 'Equity' in col:
            colRets = np.log(df[col]/df[col].shift(1))
            rets[col.split('__')[0] + " Returns"] = colRets
            i += 1

    return rets


def station2_clients(df):
    clients = df['Client Data']
    
    
    print("Mean Risk Profile of Clients = ", clients['risk_profile'].mean())

    # Assumption made of age groups
    print("Mean Age of Clients = ", clients['age_group'].mean()*18)
    
    # Risk appetite decreases as age increases
    #clients['age_group'].plot(kind='bar')

    #plt.show()
    
    # Select Relevant Features
    data = []
    data.append(clients['client_ID'])
    data.append(clients['risk_profile'])
    
    cDf = pd.DataFrame(data)
    cDf = cDf.T
    return cDf

def station2_indicators(df):
    indicators = df['Economic Data']
    monthlyInd = indicators[0]
    quarterlyInd = indicators[1]
    
    mData = []
    mData.append(monthlyInd['Consumer Sentiment Index'])
    mData.append(monthlyInd['CPI, TD-MI Inflation Gauge Idx (%m/m)'])
    mData.append(monthlyInd['AiG PMI Index'])

    qData = []
    qData.append(quarterlyInd['CPI (%q/q)'])
    qData.append(quarterlyInd['PPI (%q/q)'])
    qData.append(quarterlyInd['Company Gross Operating Profits (%q/q)'])


    mDf = pd.DataFrame(mData)
    mDf = mDf.T   
    
    qDf = pd.DataFrame(qData)
    qDf = qDf.T

    # Inflation 
    mDf['CPI, TD-MI Inflation Gauge Idx (%m/m)'].plot(title = "Monthly % Change CPI")
    plt.xlabel("Time")
    plt.ylabel('% Change CPI')
    plt.show()
    
    qDf['CPI (%q/q)'].plot(title = "Quarterly % Change CPI")
    plt.xlabel("Time")
    plt.ylabel('% Change CPI')
    plt.show()
    
    qDf['PPI (%q/q)'].plot(title = "Quarterly % Change PPI")
    plt.xlabel("Time")
    plt.ylabel('% Change PPI')
    plt.show()
    
    # Consumer Confidence
    mDf['Consumer Sentiment Index'].plot(title="Consumer Sentiment Index")
    plt.xlabel("Time")
    plt.show()
    
    # Overall business and Industry indicators
    
    return (mDf, qDf)

def station2_features(df):
    
    features = {
        'Returns': station2_returns(df['Equity Data']),
        'Clients': station2_clients(df),
        'Monthly Indicators': station2_indicators(df)[0],
        'Quarterly Indicators': station2_indicators(df)[0]
    }
    
    return features


if __name__ == '__main__':
    df = s1.station1_ETL()
    f = station2_features(df)
    print(f['Monthly Indicators'])
    

    
    