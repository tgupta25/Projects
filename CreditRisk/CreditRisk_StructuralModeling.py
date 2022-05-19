#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:56:33 2022

@author: tushar
"""
# =============================================================================
# Installing relevant libraries for the project.

!pip install sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import random
from io import StringIO
from scipy.stats import norm
from scipy.optimize import *
from sympy import *
import plotly.express as px
from tqdm import tqdm
import math
!pip install plotnine
from plotnine import ggplot, aes, geom_line, geom_bar, geom_point

# Ignoring warnings.
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Reading daily stock details from the CRSP dataset.

dsf_sample = pd.read_csv("T:\Study\MFin\Assignment 5\dsf_new.csv",nrows = 30000000,usecols=['DATE','CUSIP','PRC','SHROUT','RET'])

dsf_sample['DATE']=dsf_sample['DATE'].astype(str)
dsf_sample['YEAR']=pd.DatetimeIndex(dsf_sample['DATE']).year #Extracting YEAR as a separate column in the dataframe.
dsf_sample['CUSIP']=dsf_sample['CUSIP'].str[:6] #Constructing CUSIP as only the first 6 digits of the identifier.
dsf_sample['SHROUT']=dsf_sample['SHROUT']*1000 #Daily shares outstanding.
dsf_sample['MKT_CAP']=abs(dsf_sample['PRC'])*dsf_sample['SHROUT'] #Constructing Daily Market Capitalization of firms. 

#Filtering data on the required years.
dsf_sample = dsf_sample[(dsf_sample['YEAR']<=2020) & (dsf_sample['YEAR']>=1970)]

#Sorting data according to dates in an ascending fashion.
dsf_sample = dsf_sample.sort_values('DATE',ascending=True)
dsf_sample.reset_index(inplace=True)

#Sampling 1000 firms every year for the dataset and storing in a separate data frame.
dsf_sample_CUSIP = dsf_sample.groupby(by=['YEAR']).apply(lambda x: x[x['CUSIP'].isin(pd.Series(x['CUSIP'].unique()).sample(n=min(1000,len(x['CUSIP'].unique())), replace=False, random_state=100).to_list())])
dsf_sample_CUSIP.drop(columns={'YEAR'}, inplace=True)
dsf_sample_CUSIP.reset_index(inplace=True)

dsf_sample_CUSIP.drop(columns={'level_1','index'}, inplace=True)
del dsf_sample

#Converting returns to numeric format.
dsf_sample_CUSIP=dsf_sample_CUSIP[pd.to_numeric(dsf_sample_CUSIP['RET'],errors='coerce').notnull()]
dsf_sample_CUSIP['RET']=pd.to_numeric(dsf_sample_CUSIP['RET'])

# =============================================================================
# Function for computing lags.

def computeLag(data):
    data_temp=data.reset_index()
    data_temp[data.name+'_lagged']=data_temp.groupby(by=['CUSIP']).shift(1)[data.name]
    data_temp=data_temp.set_index(['YEAR','CUSIP'])
    return(data_temp)

# =============================================================================
# Calculating lags.

#Computing annual returns.
ann_ret=dsf_sample_CUSIP.groupby(by=['YEAR','CUSIP']).apply(lambda x:np.exp(np.sum(np.log(1+x['RET'])))-1)
ann_ret.name="RET"
ann_ret = computeLag(data=ann_ret)

#Computing annualized standard deviation of returns.
sigmae=dsf_sample_CUSIP.groupby(by=['YEAR','CUSIP'])['RET'].std()*np.sqrt(250)
sigmae.name='std_e'
sigmae=computeLag(data=sigmae)

#Computing annual equity of a firm.
E=dsf_sample_CUSIP.groupby(by=['YEAR','CUSIP'])['MKT_CAP'].first()
E.name='Equity'
E=computeLag(data=E)

# =============================================================================
# Reading annual accounting data from the COMPUSTAT dataset.

funda_df = pd.read_csv("T:\\Study\\MFin\\Assignment 4\\funda_MFI.csv", usecols=['fyear','indfmt','datafmt','popsrc','fic','consol','cusip','dlc','dltt'])

#Applying relevant filters to data.
funda_df=funda_df[funda_df['indfmt']=='INDL']
funda_df=funda_df[funda_df['datafmt']=='STD']
funda_df=funda_df[funda_df['popsrc']=='D']
funda_df=funda_df[funda_df['fic']=='USA']
funda_df=funda_df[funda_df['consol']=='C']
funda_df = funda_df[(funda_df['fyear']>=1970) & (funda_df['fyear']<=2020)]

#Setting up required variables.
funda_df['cusip']=funda_df['cusip'].str[:6]
funda_df['dlc']=funda_df['dlc']*1000000
funda_df['dltt']=funda_df['dltt']*1000000
funda_df['debt_face_value']=funda_df['dlc']+0.5*funda_df['dltt']
funda_df = funda_df[funda_df['debt_face_value']!=0] #Filtering out 0 face value of debt from the dataset.

#Computing lagged debt face value for firms.
F = funda_df.groupby(by=['fyear','cusip'])['debt_face_value'].first()
F.name="debt_face_value"
F_temp = F.reset_index()
F_temp["debt_face_value_lagged"]=F_temp.groupby(by=['cusip']).shift(1)['debt_face_value']
F = F_temp.set_index(['fyear','cusip'])

# =============================================================================
# Importing risk-free interest rate data.

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DTB3&scale=left&cosd=1970-01-01&coed=2020-12-31&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2021-11-06&revision_date=2021-11-06&nd=1954-01-04"
r = requests.get(url)
DAILYFED = pd.read_csv(StringIO(r.text))

#Wrangling DAILYFED data.
DAILYFED['YEAR']=pd.DatetimeIndex(DAILYFED['DATE']).year #Extracting YEAR as a column from DATE in DAILYFED.

#Converting 3-month treasury yields to numeric values.
DAILYFED=DAILYFED[pd.to_numeric(DAILYFED['DTB3'],errors='coerce').notnull()]
DAILYFED['DTB3']=pd.to_numeric(DAILYFED['DTB3'])
DAILYFED['r']=np.log(1+DAILYFED['DTB3']/100)

#Annualizing the data.
DAILYFED=DAILYFED.groupby(by=['YEAR'])['r'].first()#Assuming this interest rate remains same for the entire year.

# =============================================================================
# Combining datasets.

comb_data = pd.concat([ann_ret,sigmae,E,F],axis=1)
comb_data = comb_data.dropna()

firsts = comb_data.index.get_level_values(level=0)
comb_data['r'] = DAILYFED.loc[firsts].values
    
# =============================================================================
# Method 1: KMV Model - Naive Computation.

#computing firm value.
comb_data['firm_value']=comb_data['Equity']+comb_data['debt_face_value']
comb_data['firm_value_lagged']=comb_data['Equity_lagged']+comb_data['debt_face_value_lagged']

def KMV_NaiveComputation(data, sigma_d_eqn, T=1):
    if sigma_d_eqn=="0.05+0.25*sigma_e":
        data['std_d_lagged']=0.05+0.25*data['std_e_lagged']
    if sigma_d_eqn=="0.05+0.5*sigma_e":
        data['std_d_lagged']=0.05+0.5*data['std_e_lagged']
    if sigma_d_eqn=="0.25*sigma_e":
        data['std_d_lagged']=0.25*data['std_e_lagged']        
    data['std_v_lagged']=(data['Equity']/data['firm_value'])*data['std_e_lagged'] + (data['debt_face_value']/data['firm_value'])*data['std_d_lagged']
    data['dd_naive']=(np.log(data['firm_value']/data['debt_face_value']) + (data['RET_lagged'] - (data['std_v_lagged']**2)/2)*T)/(data['std_v_lagged']*np.sqrt(T))
    data['pd_naive']= 1 - norm.cdf((np.log(data['firm_value']/data['debt_face_value']) + (data['r'] - data['std_v_lagged']**2/2)*T)/(data['std_v_lagged']*np.sqrt(T)))
    return(data)

comb_data = KMV_NaiveComputation(data = comb_data, sigma_d_eqn="0.05+0.5*sigma_e")
comb_data.head()

# =============================================================================
# Method 2: KMV Model - Direct Solving Method.

V_method2 = []
sigma_V_method2 = []
T=1
for i in range(len(comb_data)):
    def myFunction(z):
        x=z[0]
        y=z[1]
        F=np.zeros(2)
        F[0] = x*norm.cdf((np.log(x/comb_data.iloc[i,6]) + (comb_data.iloc[i,8] + y**2/2)*T)/(y*np.sqrt(T))) - np.exp(-comb_data.iloc[i,8]*T)*comb_data.iloc[i,6]*norm.cdf(((np.log(x/comb_data.iloc[i,6]) + (comb_data.iloc[i,8] + y**2/2)*T)/(y*np.sqrt(T))) - y*np.sqrt(T)) - comb_data.iloc[i,4]
        F[1] = (x*norm.cdf((np.log(x/comb_data.iloc[i,6]) + (comb_data.iloc[i,8] + y**2/2)*T)/(y*np.sqrt(T)))*y)/comb_data.iloc[i,4] - comb_data.iloc[i,3]
        return (F)
    V_method2.append(fsolve(myFunction, [1000000000,0.15])[0])
    sigma_V_method2.append(fsolve(myFunction, [1000000000,0.15])[1])
    
comb_data['V_method2'] = V_method2
comb_data['sigma_V_method2'] = sigma_V_method2
del V_method2, sigma_V_method2 #Deleting to save memory.

comb_data['dd_directsolving'] = (np.log(comb_data['V_method2']/comb_data['debt_face_value']) + (comb_data['RET_lagged'] - (comb_data['sigma_V_method2']**2)/2)*T)/(comb_data['sigma_V_method2']*np.sqrt(T))
comb_data['pd_directsolving'] = 1 - norm.cdf((np.log(comb_data['V_method2']/comb_data['debt_face_value']) + (comb_data['r']  - comb_data['sigma_V_method2']**2/2)*T)/(comb_data['sigma_V_method2']*np.sqrt(T)))
comb_data.rename(columns={'sigma_V_method2' : 'sigma_V_lagged_method2'}, inplace=True)
 
# =============================================================================
# Method 3: KMV Model - Iterative Method.

comb_data.reset_index(inplace=True)
comb_data.rename(columns={'level_0' : 'YEAR','level_1' : 'CUSIP'},inplace=True)

#Sampling of 250 firms for method 3 from the sample of firms used for Method 1 and Method2.
comb_data_Method3 = comb_data.groupby(by=['YEAR']).apply(lambda x: x[x['CUSIP'].isin(pd.Series(x['CUSIP'].unique()).sample(n=min(250,len(x['CUSIP'].unique())), replace=False, random_state=100).to_list())])

#Creating a pandel data structure with YEAR, CUSIP as the indexes.
comb_data_Method3.drop(columns={'YEAR'}, inplace=True)
comb_data_Method3.reset_index(inplace=True)
comb_data_Method3.drop(columns={'level_1'}, inplace=True)
comb_data_Method3.set_index(['YEAR','CUSIP'], inplace=True)
comb_data_Method3.drop(columns={'dd_naive','V_method2','sigma_V_lagged_method2','dd_directsolving','std_d_lagged','std_v_lagged'},inplace=True)
comb_data.set_index(['YEAR','CUSIP'], inplace=True)

dsf_sample_CUSIP.set_index(['YEAR','CUSIP'], inplace=True)

#Filtering in the sampled 250 CUSIPs in the daily dataset.
indices = comb_data_Method3.index
dsf_sample_CUSIP_method3 = dsf_sample_CUSIP.loc[indices]

dsf_sample_CUSIP_method3.rename(columns={'RET':'RET_Daily'}, inplace=True)
dsf_sample_CUSIP_method3 = pd.concat([dsf_sample_CUSIP_method3, comb_data_Method3],axis=1)
dsf_sample_CUSIP_method3.drop(columns={'firm_value','firm_value_lagged','pd_naive','pd_directsolving'},inplace=True)

# =============================================================================
# Setting up a function to iteratively calculate the best estimate of firm value, std. deviation of firm value.

#Renaming columns.
dsf_sample_CUSIP_method3.rename(columns={col: col+'_annual' for col in dsf_sample_CUSIP_method3.columns[5:]}, inplace=True)

dsf_sample_CUSIP_method3['sigma_V_lagged_estimate'] = dsf_sample_CUSIP_method3['std_e_lagged_annual'] #Setting the initial estimate of std. deviation of the firm's value as the std. deviation of equity.
list_t = dsf_sample_CUSIP_method3.values.tolist() #Converting all the values of the dataframe to a list for faster computation.

#Taking annualized volatility from daily stock prices as an initial estimate of the volatility of the firm's value for the Iterative method.
def KMV_IterativeMethod(data):
    V_estimate=[]
    for i in tqdm(range(len(data))):
        def mysolveV(z):
            x  = z
            F = 0
            F = x*norm.cdf((np.log(x/data[i][11]) + (data[i][13] +data[i][14]**2/2)*1)/(data[i][14]*1)) - np.exp(-data[i][13]*1)*data[i][11]*norm.cdf((np.log(x/data[i][11]) + (data[i][13] - data[i][14]**2/2)*1)/(data[i][14]*1)) -data[i][4]
            return(F)z
        V_estimate.append(fsolve(mysolveV, 100000000))
    
    data = pd.DataFrame(data)
    data.columns = dsf_sample_CUSIP_method3.columns
    data.index = dsf_sample_CUSIP_method3.index
    data['V_estimate'] = V_estimate
    data['RET_V_lagged_estimate'] = data.groupby(level=['CUSIP','YEAR'])['V_estimate'].apply(lambda x: (x - x.shift(1))/x.shift(1))
    data['sigma_V_lagged_estimate_new'] = data.groupby(level = ['CUSIP','YEAR'])['RET_V_lagged_estimate'].std()*np.sqrt(250) #Computing annualized standard deviation of returns of the firm's value.
    if ((data['sigma_V_lagged_estimate_new'] - data['sigma_V_lagged_estimate']).all() > 0.01):
        data['sigma_V_lagged_estimate'] = data['sigma_V_lagged_estimate_new']
        list_t = data.values.to_list()
        return(KMV_IterativeMethod(data = list_t))
    else:
        return(data)
        
dsf_sample_CUSIP_method3 = KMV_IterativeMethod(data=list_t)

#Computing DD, PD using Method 3 - Iterative Method.
dsf_sample_CUSIP_method3['dd_iterative'] = (np.log(dsf_sample_CUSIP_method3['V_estimate']/dsf_sample_CUSIP_method3['debt_face_value_annual']) + (dsf_sample_CUSIP_method3['RET_lagged_annual'] - (dsf_sample_CUSIP_method3['sigma_V_lagged_estimate_new']**2)/2)*T)/(dsf_sample_CUSIP_method3['sigma_V_lagged_estimate_new']*np.sqrt(T))
dsf_sample_CUSIP_method3['pd_iterative'] = 1 - norm.cdf((np.log(dsf_sample_CUSIP_method3['V_estimate']/dsf_sample_CUSIP_method3['debt_face_value']) + (dsf_sample_CUSIP_method3['r_annual']  - comb_data['sigma_V_lagged_estimate_new']**2/2)*T)/(dsf_sample_CUSIP_method3['sigma_V_lagged_estimate_new']*np.sqrt(T)))

dsf_sample_CUSIP_method3 = dsf_sample_CUSIP_method3.groupby(by=['YEAR','CUSIP']).mean().loc[:,['V_estimate','sigma_V_lagged_estimate','dd_iterative','pd_iterative']]

# =============================================================================
# Generating descriptive statistics for DD, PD measures.

comb_data.loc[:,['dd_naive','pd_naive','dd_directsolving','pd_directsolving']].describe()
dsf_sample_CUSIP_method3.loc[:,['dd_iterative','pd_iterative']].describe()

# =============================================================================
# Capturing the correlation between DD, PD measures computed using different methods.

comb_data_corr = pd.concat([comb_data.loc[:,['dd_naive','pd_naive','dd_directsolving','pd_directsolving']],dsf_sample_CUSIP_method3.loc[:,['dd_iterative','pd_iterative']]],axis=0)
comb_data_corr.fillna(0, inplace=True)
comb_data_corr.corr()

#Function for computing descriptive statistics.
def computeStats(data, attribute):
    data = pd.concat([data.groupby(level='YEAR')[attribute].mean(), data.groupby(level='YEAR')[attribute].quantile(0.25), data.groupby(level='YEAR')[attribute].quantile(0.5),data.groupby(level='YEAR')[attribute].quantile(0.75)],axis=1)
    data.columns.values[0]=attribute+'_mean'
    data.columns.values[1]=attribute+'_.25'
    data.columns.values[2]=attribute+'_.50'
    data.columns.values[3]=attribute+'_.75' 
    return(data)
    
comb_data_stats_method1_dd = computeStats(data=comb_data, attribute='dd_naive')
comb_data_stats_method1_pd = computeStats(data=comb_data, attribute='pd_naive')
comb_data_stats_method2_dd = computeStats(data=comb_data, attribute='dd_directsolving')
comb_data_stats_method2_pd = computeStats(data=comb_data, attribute='pd_directsolving')
comb_data_stats_method3_dd = computeStats(data=dsf_sample_CUSIP_method3, attribute='dd_iterative')
comb_data_stats_method3_pd = computeStats(data=dsf_sample_CUSIP_method3, attribute='pd_iterative')

# =============================================================================
# Plots for DD, PD estimates and their descriptive statistics with macroeconomic variables.

comb_data_stats_method1_dd.plot(title='DD:Method1')
comb_data_stats_method2_dd.plot(title='DD:Method2')
comb_data_stats_method3_dd.plot(title='DD:Method3')

# Plots with NBER recession periods.

NBER = pd.read_csv('T:\\Study\\MFin\\Assignment 10\\USREC.csv')
NBER['DATE'] = pd.DatetimeIndex(NBER['DATE']).year
NBER['USREC'] = NBER['USREC'].apply(lambda x : 500 if x!=0.000000 else 0)
NBER.set_index('DATE',inplace=True)

#Plotting functions.
ax_1_dd = comb_data_stats_method1_dd.plot(kind='line',linestyle='--')
ax_1_pd = comb_data_stats_method1_pd.plot(kind='line',linestyle='--')
NBER.plot(ax=ax_1_dd)
NBER.plot(ax=ax_1_pd)
ax_2_dd = comb_data_stats_method2_dd.plot(kind='line',linestyle='--')
ax_2_pd = comb_data_stats_method2_pd.plot(kind='line',linestyle='--')
NBER.plot(ax=ax_2_dd)
NBER.plot(ax=ax_2_pd)
ax_3_dd = comb_data_stats_method3_dd.plot(kind='line',linestyle='--')
ax_3_pd = comb_data_stats_method3_pd.plot(kind='line',linestyle='--')
NBER.plot(ax=ax_3_dd)
NBER.plot(ax=ax_3_pd)

# Plots with Moody's Fed-Fund Spread.

BAAFFM = pd.read_csv('T:\\Study\\MFin\\Assignment 10\\BAAFFM.csv')
BAAFFM['DATE'] = pd.DatetimeIndex(BAAFFM['DATE']).year
BAAFFM.set_index('DATE',inplace=True)

#Plotting functions.
print(pd.concat([comb_data_stats_method1_dd,BAAFFM],axis=0).plot())
print(pd.concat([comb_data_stats_method1_pd,BAAFFM],axis=0).plot())
print(pd.concat([comb_data_stats_method2_dd,BAAFFM],axis=0).plot())
print(pd.concat([comb_data_stats_method2_pd,BAAFFM],axis=0).plot())
print(pd.concat([comb_data_stats_method3_dd,BAAFFM],axis=0).plot())
print(pd.concat([comb_data_stats_method3_pd,BAAFFM],axis=0).plot())

# Plots with Cleveland Financial Stress Index.

CFSI = pd.read_csv('T:\\Study\\MFin\\Assignment 10\\CFSI.csv')
CFSI['DATE'] = pd.DatetimeIndex(CFSI['DATE']).year
CFSI.set_index('DATE',inplace=True)

#Plotting functions.
print(pd.concat([comb_data_stats_method1_dd,CFSI],axis=0).plot())
print(pd.concat([comb_data_stats_method1_pd,CFSI],axis=0).plot())
print(pd.concat([comb_data_stats_method2_dd,CFSI],axis=0).plot())
print(pd.concat([comb_data_stats_method2_pd,CFSI],axis=0).plot())
print(pd.concat([comb_data_stats_method3_dd,CFSI],axis=0).plot())
print(pd.concat([comb_data_stats_method3_pd,CFSI],axis=0).plot())
