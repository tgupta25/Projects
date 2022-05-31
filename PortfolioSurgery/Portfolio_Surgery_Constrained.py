#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:14:09 2022

@author: tushar
"""

#Declaring a class for constrained linear regression.
from sklearn.linear_model._base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y

class ConstrainedLinearRegression(LinearModel, RegressorMixin):

    def __init__(self, A, B, fit_intercept=True, normalize=False, copy_X=True, tol=1e-15, lr=1.0):
        self.A = A
        self.B = B
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.lr = lr

    def fit(self, X, y, initial_beta=None):
        X, y = check_X_y(
            X, y, 
            accept_sparse=['csr', 'csc', 'coo'], 
            y_numeric=True, multi_output=False
        )
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y,
            fit_intercept=self.fit_intercept, 
            normalize=self.normalize, 
            copy=self.copy_X
        )
        if initial_beta is not None:
            # providing initial_beta may be useful, 
            # if initial solution does not respect the constraints. 
            beta = initial_beta
        else:
            beta = np.zeros(X.shape[1]).astype(float)
        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        while not (np.abs(prev_beta - beta)<self.tol).all():
            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X,beta) - y, X)
                max_coef = np.inf
                min_coef = -np.inf
                for a_row, b_value in zip(self.A, self.B):
                    if a_row[i] == 0:
                        continue
                    zero_out = beta.copy()
                    zero_out[i] = 0
                    bound = (b_value - np.dot(zero_out, a_row)) / a_row[i] 
                    if a_row[i] > 0:
                        max_coef = np.minimum(max_coef, bound)
                    elif a_row[i] < 0:
                        min_coef = np.maximum(min_coef, bound)
                assert min_coef <= max_coef, "the constraints are inconsistent"
                beta[i] = np.minimum(
                    max_coef, 
                    np.maximum(
                        min_coef,
                        beta[i] - (grad[i] / hessian[i,i]) * self.lr
                    )
                )
        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self  

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.linear_model import LinearRegression

# stock is the possible stock price space
stock = np.arange(1,101)

# Then call and put are the possible call and put payments space
call = np.maximum(stock - np.transpose([[10,30,50,70,90]]), 0)
put = np.maximum(-stock + np.transpose([[10,30,50,70,90]]), 0)

# import portfolio data
# use your own file path to replace
file_path = "Portfolio Surgery Test 1-1.xlsx"
portfolio_dataframe = pd.read_excel(file_path, sheet_name="Tab 1")

#Plotting portfolio returns.
plt.plot(portfolio_dataframe['Portfolio'])
plt.title('Portfolio')
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.show()

# plot call and put
call, put = np.transpose(call), np.transpose(put)

plt.plot(call)
plt.title('call')
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.show()

plt.plot(put)
plt.title('put')
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.show()

# x is the independent variable, axis = 1 means concat 
x = np.concatenate((call, put), axis=1)

negative_portfolio1 = np.minimum(portfolio_dataframe['Portfolio'], 0)
plt.plot(negative_portfolio1)
plt.title("negative part of portfolio1")
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.show()

#Importing call, put prices.
prices = pd.read_excel(file_path, sheet_name="Tab 2")
prices = prices.loc[prices['Strikes'].isin([10,30,50,70,90])]
prices = np.concatenate((list(prices['Call Prices']), list(prices['Put Prices']))).reshape(1,-1)

#Budget.
budget = [150000]

#Constrained Linear Regression.
model = ConstrainedLinearRegression(    
    A = prices,
    B = budget, 
    fit_intercept=False, 
    lr=0.5)
fitted_model = model.fit(x, negative_portfolio1.values)
beta=fitted_model.coef_

#After surgery.
aftersurgery = negative_portfolio1 - np.dot(x,beta)

#Comparison of payoffs before and after surgery.
plt.plot(stock,portfolio_dataframe['Portfolio'])
plt.plot(stock, aftersurgery)
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.legend(["Portfolio","after_surgery"])
plt.title("comparison")
plt.show()

# A closer look at the portfolio payoff after surgery.
plt.plot(stock, aftersurgery)
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.legend(["after_surgery"])
plt.show()

#Comparison of original portfolio's payoff and option payoff (used for getting rid of the negative payoff).
option_payoff = np.dot(x,beta)
plt.xlabel('stock price')
plt.ylabel('payoff')
plt.plot(option_payoff)
plt.plot(portfolio_dataframe['Portfolio'])
plt.legend(["combined options payoff","Original Portfolio"])
plt.show()

