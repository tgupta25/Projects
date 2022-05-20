#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:50:16 2022

@author: tushar
"""
#Installing relevant librariies for the project.
!pip install sec_edgar_downloader
!pip install secedgar
!pip install zipfile2
!pip install pandas_datareader
!pip install torch
!pip install transformers

!python -m pip install --upgrade pip transformers

#Importing libraries for the peoject.
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit, fsolve, root
from matplotlib.backends.backend_pdf import PdfPages
import sec_edgar_downloader
from sec_edgar_downloader import Downloader
import secedgar
import secedgar.filings
from secedgar.filings import Filing, FilingType
import urllib3
import urllib.request
import lib
import zipfile
import zipfile2

from zipfile import ZipFile
import time
from urllib.request import urlretrieve
import re
import os
import os.path
from os.path import isfile, join
import requests
from pathlib import Path
import glob
import nltk.sentiment.vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print(os.getcwd())

# Below code chunk downloads the master zipped file for every quarter from 1995 to 2020 and unzipping all the files in a separate folder.

hdr = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36'}
allquarters = ('QTR1', 'QTR2', 'QTR3', 'QTR4')
#Reading in SEC Data.
for year in range(1995,2021):
    for quarter in allquarters:
        url = 'https://www.sec.gov/Archives/edgar/full-index/'+str(year)+'/'+str(quarter)+'/master'+'.zip'
        print(url)
        file_path = './data/master_zipped/'+str(year)+'_'+str(quarter)+'.zip'
        req = requests.get(url,headers=hdr, stream=True)
        if req.ok:
            with open(file_path,'wb') as f:
                for chunk in req.iter_content(chunk_size=1024*8):
                    if chunk:
                        f.write(chunk)
                        f.flush()                      
            
        with zipfile2.ZipFile('./data/master_zipped/'+str(year)+'_'+str(quarter)+'.zip', 'r') as zip_ref:
            zip_ref.extractall('./data/master_unzipped/'+str(year)+'_'+str(quarter))

np_data = np.empty((0, 5), str)
for year in range(1995, 2021):
    print(year)
    for quarter in allquarters:
        with open('./data/master_unzipped/' + str(year) + '_' + str(quarter) + '/master.idx', encoding='latin-1') as z:
            for i in range(11):
                z.readline()
            records = []
            for line in z:
                    line = line[0:-1]
                    temp_list = list(line.split('|'))
                    if (temp_list[2] == '8-K'):
                        records.append(temp_list)
            np_records = np.array(records)
            idx = np.random.randint(np.size(np_records, 0), size = 100)
            np_data = np.append(np_data, np_records[idx, :], axis = 0)
            
df_filelist = pd.DataFrame(np_data, columns = ['CIR', 'CompanyName', 'FormTypes', 'DATE', 'link'])
df_filelist.to_csv('filelist.csv')

#Combining the above file list with DSF and COMPUSTAT datasets.

#Reading the COMPUSTAT dataset.
funda_df = pd.read_csv('./data/funda_MFI.csv', header = 0)
funda_df['cusip'] = funda_df['cusip'].str[0:6]
funda_df = funda_df.loc[funda_df['fyear'] >= 1995] #Keeping only years of interest in the data.
funda_df = funda_df.loc[funda_df['fyear'] <= 2020] #Keeping only years of interest in the data.
funda_df = funda_df.loc[:,['fyear','cusip','cik']]
funda_df = funda_df.set_axis(['FYEAR', 'CUSIP', 'CIK'], axis = 1,inplace = False)

# Merging funda_df and filelist here.
CIK = pd.read_csv('./filelist.csv', header = 0)
CIK = CIK.iloc[:, 1:5]
CIK['DATE'] = pd.to_datetime(CIK['DATE'], format='%Y/%m/%d')
CIK['FYEAR'] = pd.DatetimeIndex(CIK['DATE']).year
CIK = CIK.set_axis(['CIK', 'CompanyName', 'FormTypes', 'filingdate', 'FYEAR'], axis = 1,inplace = False)
CIK[["CIK"]] = CIK[["CIK"]].apply(pd.to_numeric, errors = 'coerce')
funda_df_CIK = pd.merge(funda_df, CIK, how = 'inner', on = ['FYEAR', 'CIK'])
funda_df_CIK = funda_df_CIK.loc[:,['CUSIP', 'FYEAR', 'CIK', 'CompanyName', 'FormTypes', 'filingdate']]

# Exporting to csv.
funda_df_CIK.to_csv('./funda_df_CIK_final.csv')

# Merging DSF and COMPUSTAT, CIK data here.

#Reading CRSP DSF file.
dsf = pd.read_csv('./data/dsf_new.csv', header = 0,nrows=50000)
dsf['DATE'] = pd.to_datetime(dsf['DATE'], format='%Y/%m/%d')
dsf['FYEAR'] = pd.DatetimeIndex(dsf['DATE']).year
dsf['FYEARmin1'] = dsf['FYEAR'] - 1
dsf['CUSIP'] = dsf['CUSIP'].str[0:6]

# Combining all the 3 datasets.
comb_data = pd.merge(dsf, funda_df_CIK, how = 'inner', left_on = ['CUSIP', 'FYEAR'], right_on = ['CUSIP', 'FYEAR'])

# Exporting the final combined dataset to a csv file for future use and reference.
comb_data.to_csv('./comb_data_final.csv')

# Step 2: Event Studies
comb_data = comb_data.iloc[:,1:]
date_condition = [(comb_data['DATE'] == comb_data['filingdate']), (comb_data['DATE'] != comb_data['filingdate'])]
date_values = [1, 0]
comb_data['filingdatefound'] = np.select(date_condition, date_values)
comb_data[["RET"]] = comb_data[["RET"]].apply(pd.to_numeric, errors = 'coerce')

# Calcualting Cumuative Abnormal Returns (CAR).
C = 2.55 * (10 ** (-6))
comb_data['TURNOVER'] = comb_data['VOL']/comb_data['SHROUT']
comb_data['ADJUSTEDTO'] = (np.log(comb_data['TURNOVER'] + C))

comb_data['rollreg'] = comb_data.groupby('CUSIP')['ADJUSTEDTO'].shift(-11).rolling(60).sum()
comb_data['ATOmean'] = comb_data['rollreg']/60

comb_data['ADJVALUES'] = np.power((comb_data['ADJUSTEDTO'] - comb_data['ATOmean']), 2)
comb_data['std'] = comb_data.groupby('CUSIP')['ADJVALUES'].shift(-11).rolling(60).sum()

comb_data['ATOstd'] = np.sqrt(comb_data['std']/60)
comb_data['ATO'] = ((comb_data['ADJUSTEDTO'] - comb_data['ATOmean'])/(comb_data['ATOstd']))
comb_data = comb_data.drop(columns=['TURNOVER', 'ADJUSTEDTO', 'rollreg', 'ATOmean', 'ADJVALUES', 'std', 'ATOstd'])

# Below code chunk deals with setting up regression to calculate alpha, beta for each CUSIP.
def ols_res(df, xcols,  ycol):
    return sm.OLS(df[ycol], df[xcols]).fit().predict()

linresults = comb_data.groupby('CUSIP').apply(ols_res, 'VWRETD', 'RET')

linCUSIPAB = pd.DataFrame(linresults.copy())
cusipAB = linCUSIPAB.iloc[:, 0:1]
cusipAB = cusipAB.reset_index()
cusipAB = cusipAB.iloc[:, 0:1]
cusipAB = cusipAB.set_axis(['CUSIP'], axis = 1,inplace = False)
n = cusipAB['CUSIP'].count()

coefs_array = np.array(linresults)
coefs_list = coefs_array.tolist()
coefs_values_df = pd.DataFrame(np.concatenate(coefs_list))

alphasV = coefs_values_df.iloc[0:n, 0:1]
alphasV = alphasV.reset_index()
betasV = coefs_values_df.iloc[n:2*n, 0:1]
betasV = betasV.reset_index()
ab = pd.concat([alphasV, betasV], axis = 1)
abCUSIP = pd.concat([cusipAB, ab], axis = 1)
abCUSIP = abCUSIP.set_axis(['CUSIP', 'index1', 'alpha', 'index2', 'beta'], axis = 1,inplace = False)
abCUSIP = abCUSIP.drop(columns=['index1', 'index2'])
comb_data = pd.merge(df, abCUSIP, how = 'inner', on = 'CUSIP')

# Below code chunk deals with calculating cumulative abnormal returns for different windows around the 8-k filing dates. 

#Cumulative Abnormal Returns.
comb_data['AR'] = comb_data['RET'] - (comb_data['alpha'] + comb_data['beta']*comb_data['VWRETD'])
comb_data['CAR0'] = comb_data['AR']
comb_data['CAR1'] = comb_data.groupby(['CUSIP'])['AR'].shift(1).rolling(3).sum().shift(-1)
comb_data['CAR2'] = comb_data.groupby(['CUSIP'])['AR'].shift(2).rolling(5).sum().shift(-2)
comb_data['CAR3'] = comb_data.groupby(['CUSIP'])['AR'].shift(3).rolling(7).sum().shift(-3)
comb_data['CAR5'] = comb_data.groupby(['CUSIP'])['AR'].shift(5).rolling(11).sum().shift(-5)

#Descriptive statistics.
step2values = comb_data.loc[comb_data['filingdatefound'] == 1]
ttestDATA = step2values[['CIK'] + ['ATO'] + ['CAR0'] + ['CAR1'] + ['CAR2'] + ['CAR3'] + ['CAR5']]
ttestDATA.describe()

def ttest(data, i):
    tstat = (data[i].mean() - 0)/((data[i].std()/np.sqrt(data[i].shape[0])))
    print('The t-stat for', i, 'is')
    print(tstat)
    if tstat > 1.64:
        print('It is statistically significant at 0.05')
        print('\n')
    else:
        print('It is not statistically significant at 0.05')
        print('\n')
    return tstat

tATO = ttest(ttestDATA, 'ATO')
tCAR0 = ttest(ttestDATA, 'CAR0')
tCAR1 = ttest(ttestDATA, 'CAR1')
tCAR2 = ttest(ttestDATA, 'CAR2')
tCAR3 = ttest(ttestDATA, 'CAR3')
tCAR5 = ttest(ttestDATA, 'CAR5')

ATOplot = ttestDATA['ATO'].plot.hist(alpha=0.5)
plt.xlabel('Value of Cumulative Abnormal Volume')
plt.title('Cumulative Abnormal Volume')
plt.show()

CAR0plot = ttestDATA['CAR0'].plot.hist(alpha=0.5)
plt.xlabel('Value of Cumulative Abnormal Returns')
plt.title('Cumulative Abnormal Returns Window of 0')
plt.show()

CAR1plot = ttestDATA['CAR1'].plot.hist(alpha=0.5)
plt.xlabel('Value of Cumulative Abnormal Returns')
plt.title('Cumulative Abnormal Returns Window of 1')
plt.show()

CAR2plot = ttestDATA['CAR2'].plot.hist(alpha=0.5)
plt.xlabel('Value of Cumulative Abnormal Returns')
plt.title('Cumulative Abnormal Returns Window of 2')
plt.show()

CAR3plot = ttestDATA['CAR3'].plot.hist(alpha=0.5)
plt.xlabel('Value of Cumulative Abnormal Returns')
plt.title('Cumulative Abnormal Returns Window of 3')
plt.show()

CAR5plot = ttestDATA['CAR5'].plot.hist(alpha=0.5)
plt.xlabel('Value of Cumulative Abnormal Returns')
plt.title('Cumulative Abnormal Returns Window of 5')
plt.show()

ttestDATA.to_csv(("./CAR.csv"))

# Step 1: Rudimentary Sentiment Analysis

np_data = pd.read_csv('./filelist.csv')
np_data = np_data.iloc[:, 1:]

cikREADS = pd.read_csv("./comb_data_final.csv")
cikREADS = cikREADS.iloc[:, 10:11]
cikREADS = cikREADS.drop_duplicates()
cikREADS = cikREADS.reset_index()
cikREADS = cikREADS.iloc[:, 1:]
cikREADS = cikREADS.set_axis(['CIR'], axis = 1,inplace = False)

np_data = pd.merge(np_data, cikREADS, how = 'inner', on = 'CIR')

np_data.to_csv(("./CIKvalsMERGE.csv"))

files_8_K = list(np_data.iloc[:, 4])
for f in files_8_K:
    time.sleep(0.15)
    url = 'https://www.sec.gov/Archives/' + f
    temp_list = list(f.split('/'))
    urllib.request.urlretrieve(url, './data/8_K_files/' + temp_list[2] + '_' + temp_list[3])
    
# Post Txt downloads

masterdictionary = pd.read_csv('./data/LoughranMcDonald_MasterDictionary_2020.csv')
masterdictionary[["Positive"]] = masterdictionary[["Positive"]].apply(pd.to_numeric, errors = 'coerce')
masterdictionary[["Negative"]] = masterdictionary[["Negative"]].apply(pd.to_numeric, errors = 'coerce')

regex_words = r"(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+"
def score_words(pos, neg, listfiles):
    tonevalues = []
    for file in listfiles:
        # with open('./data/8_K_files/'+str(file)) as funcFILE:
        with open(file) as funcFILE:
            txt = funcFILE.read()
            txt = txt.upper()
            txt = re.findall(regex_words, txt)
            negative_score = 0
            positive_score = 0
            for word in neg:
                negative_score += txt.count(word)
            for word in pos:
                positive_score += txt.count(word)
            tone = (positive_score - negative_score)/len(txt)
            tonevalues.append((file, tone))
    tone_df = pd.DataFrame(tonevalues)
    tone_df = tone_df.set_axis(['FileName', 'tonescore'], axis = 1,inplace = False)
    return tone_df

MD = masterdictionary
negatives = set(MD[MD['Negative'] != 0]['Word'])
positives = set(MD[MD['Positive'] != 0]['Word'])

func_listfiles = glob.glob("./data/8_K_files/*.txt")
filescores = score_words(positives, negatives, func_listfiles)

filescores.describe()

filescores.to_csv(("./filescores_words.csv"))

# Step 2 Parsing

finalwords = pd.read_csv("./wordbase.csv")
finalwords = finalwords.iloc[:, 1:]

finalwords = finalwords.set_axis(['CIK', 'ATO', 'CAR0', 'CAR1', 'CAR2', 'CAR3', 'CAR5', 'FileNameFull',
       'tonescore', 'filebind', 'CompanyName', 'FormTypes', 'DATE', 'quintile'], axis = 1, inplace = False)
filescores = pd.read_csv("/filescores_words.csv")
filescores.iloc[:, 1:]
filescores['filebind'] = filescores['FileName']
filescores['filebind'] = filescores['filebind'].str[24:]
allfiles_sentences = pd.merge(finalwords, filescores, how = 'inner', on = 'filebind')

#Trimming down data to save computation time.
sentencesfiles_df = allfiles_sentences['FileName'].drop_duplicates()
sentencesfiles_df = pd.DataFrame(sentencesfiles_df)
sentencesfiles_df = sentencesfiles_df.reset_index()
sentencesfiles_df = sentencesfiles_df.iloc[:, 1:]
sentencesfiles_list = sentencesfiles_df['FileName'].to_list()

# regex_sentences = r"[“’“]?(A-Z)(((Mr|Ms|Mrs|Dr|Capt|Col)\.\s+((?!\w{2,}[.?!][’”]?\s+[“’]?[A-Z]).))?)((?![.?!][“’]?\s+[“’]?[A-Z]).)[.?!]+[“’”]?"

def score_sentences(listfiles):
    tonevalues_sentences = []
    for file in listfiles:
        with open(file) as funcFILE:
            txt = funcFILE.read()
            txt = txt.upper()
            phrases = sent_tokenize(txt)
            sid = SentimentIntensityAnalyzer()
            i = 0
            j = 0
            for sentence in phrases:
                i = i + 1
                ss = sid.polarity_scores(sentence)['compound']
                j = j + ss
            score = j/i
            tonevalues_sentences.append((file, score))
    tonesentences_df = pd.DataFrame(tonevalues_sentences)
    tonesentences_df = tonesentences_df.set_axis(['FileName', 'tonescore_sentence'], axis = 1,inplace = False)
    return tonesentences_df

sentecesdf = score_sentences(sentencesfiles_list)
sentecesdf.to_csv(("./filescores_sentences.csv"))


