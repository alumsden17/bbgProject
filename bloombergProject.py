import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.config_init import max_colwidth_doc
import sys
import seaborn as sns
#fx_df = pd.read_excel("Alec Lumsden\Downloads\DBG Data Set Presentation Prep Doc.xlsx")




class timeSeries(pd.DataFrame):
    def __init__(self, df):
        self.df = df

    def correlation_matrix(self):

       return None
    def momentum(self,freq,col):
        #MACD(self.df)
        RSI(self.df,freq,col)
        MACD(self.df,col)
    def hedge_ratios(self,x):
        x = self.df[x].pct_change().tolist()
        x[0] =0
        points = {}
        points['dataPoints']  = []
        points['HR'] = []
        for col in  self.df.columns:
            if col  == 'date' or col == x:
                continue
            points['dataPoints']+=[col]
            y = self.df[col].pct_change().tolist()
            y[0] =0
            points['HR'] +=[round(np.corrcoef(x,y)[0][1] *(np.std(y)/np.std(x)),2)]
        return pd.DataFrame(points)




def MACD(df,title,signal=True):

    weight = 2/(13.0)
    emas = []
    prev_ema = 0
    df['ema12'] = df[title].ewm(span= 12,adjust = False).mean()
    df['ema26'] = df[title].ewm(span=26, adjust=False).mean()
    df['MACD'] =  df['ema26']-df['ema12']

    df['ema9'] =df['MACD'].ewm(span =9,adjust=False).mean()

    df['Signal'] =df['MACD']-df['ema9']
    df=df[~df['date'].isnull()]
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize = (12,6))
    ax_1 = ax.twinx()
    fig = sns.lineplot(data= df[['MACD','ema9']],ax=ax).set(title= 'MACD '+title)
    ax.set_xticklabels(labels=df['date'].tolist(),rotation= 45,ha='right')
    plt.show()
    if signal:
        fig1,ax1= plt.subplots(figsize=(10,6))
        colors = ['red'if value<0 else 'green' for value in df['Signal']]
        fig1 = sns.barplot(x='date',y='Signal',palette=colors,data=df,ax=ax1).set(title='Signal '+title)
        plt.show()




    return df

def RSI(df, freq, title ):
    df['px'] = df[title]
    df['pct_chg'] = df['px'].pct_change()
    df =df.iloc[1:]

    gains = []

    losses =[]
    upper =[]
    lower =[]
    rs_values = []
    dates =[]
    for x in range(len(df)):
        if x +(freq-1)<=len(df)-1:
            temp= df.iloc[x:x+freq]
            pos = temp[temp['pct_chg']>=0]['pct_chg']
            neg =temp[temp['pct_chg'] < 0]['pct_chg']
            if np.isnan(pos.mean()) ==True:
                rs = 100
            elif np.isnan(neg.mean())== True:
                rs =0
            else:
                rs = 100-(100/(1+(pos.mean()/(-1*neg.mean()))))
            rs_values +=[rs]
            dates +=[temp['date'].iloc[0]]
            upper+=[70]
            lower +=[30]
        else:
            break

    df = pd.DataFrame({'date':dates,'rsi':rs_values,'upper':upper,'lower':lower})
    plt.plot(df['date'],df['rsi'],label='RSI')
    plt.plot(df['date'],df['upper'],label = 'upper limit')
    plt.plot(df['date'],df['lower'],label='lower limit')
    plt.legend()
    plt.title('RSI Indicator '+title)
    plt.xlabel('date')
    plt.ylabel('RSI')
    plt.show()
    return df


def lnd(d,s):
    return d+s
#%%
def dataRange(df,start,end):

    return (df['date'] > start) & (df['date'] < end)

# =bg.timeSeries(fx_df)
#fx_df=fx_df.rename(columns={'Date':'date'})
#RSI(fx_df,32,'USDJPY')

lnd(1,2)
