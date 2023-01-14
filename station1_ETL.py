import pandas as pd
import datetime


"""
Created on Mon Jul  4 15:36:09 2022

@author: William Zheng (z5313015)
"""
    

def station1_ETL_equities(fileDB):
    """
    Receive and clean raw data
    param: fileDB - directory path of data file
    return: df - return cleaned data
    """
    df= pd.read_csv(fileDB + 'ASX200top10.csv', header = None)
    
    colNames = []
    temp = ""
    for column in df.iloc[0]:
        if isinstance(column, str):
            temp = column
        colNames.append(temp)
    
    # Add tickers to first column
    df.iloc[0] = colNames    
    
    # combine iloc[0] and iloc[1] and set to column
    df.columns = (df.iloc[0] + '__' + df.iloc[1])
    
    # Remove first row and reset all index
    df = df.iloc[2:].reset_index(drop=True)
    
    for i, col in enumerate(df):
        if i == 0:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
        else:
            df[col] = pd.to_numeric(df[col],errors = 'coerce')
            
    
    return df


def station1_ETL_clients(fileDB):
    df = pd.read_csv(fileDB + 'Client_Details.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    
    for i, col in enumerate(df):
        df[col] = pd.to_numeric(df[col],errors = 'coerce')
            
    return df
            


def station1_ETL_indicators(fileDB):
    df = pd.read_excel(fileDB + 'Economic_Indicators (1).xlsx', header = None)
    df = df[3:]
        
    
    firstEmptyRow = df[df.isnull().all(axis=1) == True].index.tolist()[0]
    
    df_quarterly = df[firstEmptyRow-2:]
    df_monthly = df[0:firstEmptyRow-3]

    df_quarterly, df_monthly = df_quarterly.T, df_monthly.T
    df_quarterly.columns, df_monthly.columns = df_quarterly.iloc[0], df_monthly.iloc[0]
    df_quarterly, df_monthly = df_quarterly[1:], df_monthly[1:]

    df_quarterly.index = df_quarterly['Quarterly Indicators']
    df_monthly.index = df_monthly['Monthly Indicators']

    df_quarterly.drop('Quarterly Indicators', inplace = True, axis=1)    
    df_monthly.drop('Monthly Indicators', inplace = True, axis=1)    


    
    for i, col in enumerate(df_monthly):
        df_monthly[col] = pd.to_numeric(df_monthly[col],errors = 'coerce')
   
    for i, col in enumerate(df_quarterly):
        df_quarterly[col] = pd.to_numeric(df_quarterly[col],errors = 'coerce')
    
    df_monthly.index = pd.to_datetime(df_monthly.index,  format="%m/%d/%Y")
    df_quarterly.index = pd.to_datetime(df_quarterly.index,  format="%m/%d/%Y")
    
    return (df_monthly, df_quarterly)


def station1_ETL_news(fileDB):
    df = pd.read_json(fileDB + 'news_dump.json')
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %b \'%y %H:%M %p')

    return df


def station1_ETL():
    fileDB = 'C:/Users/Will/OneDrive/Desktop/FINS3645/Project/Data/'
    data = {
        'Equity Data': station1_ETL_equities(fileDB),
        'Client Data': station1_ETL_clients(fileDB),
        'Economic Data': station1_ETL_indicators(fileDB),
        'News Data': station1_ETL_news(fileDB)
    }
        
    return data



if __name__ == '__main__':
    fileDB = 'C:/Users/Will/OneDrive/Desktop/FINS3645/Project/Data/'
    dfEquities = station1_ETL_equities(fileDB)
    dfClient = station1_ETL_clients(fileDB)
    dfIndicators = station1_ETL_indicators(fileDB)
    dfNews = station1_ETL_news(fileDB)
    
    #df = station1_ETL()
    #dfEq = df['Economic Data']
    #print(dfEq)

    







