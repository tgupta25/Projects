# **Credit Risk: Structural Modelling**

## **Datasets Used**
- CRSP DSF
- COMPUSTAT
- Risk-freeinterestrate(FRED)
- NBER Recession
- Moody’s Fed-Fund Spread
- Cleveland Financial Stress Index

## **Implementation Methodology**
- The CRSP DSF dataset has been used to randomly sample 1000 firms each year.
- All the data has been annualized and annual return, standard deviation, market capitalization for each security, each year has been stored in this dataframe.
- The combined dataset has been used to compute and store the Distance-to-Default, Probability-of-Default using the KMV model. Below are the 3 approaches used:
    -Naive estimation
    -Direct Solving
    -terative Solving
- For the iterative solving method, 250 firms have been randomly sampled each year and stored in a separate dataframe.
- The computed DD, PD measures using all the 3 methods have thereafter been used to compute several statistics for each of the 3 methods.
- The trend for these DD, PD measures has been studied over time along with macroeconomic indicators like NBER Recession, Moody’s BAA-Fed Fund Spread, Cleveland Financial Stress index.

## **KMV Model**

KMV Model is based on the structural approach to calculate EDF (expected default frequency), which is a forward looking measure of default probability. Below is the equation for calculating Distance-to-Default.

<p align="center">
<img width="381" alt="image" src="https://user-images.githubusercontent.com/65303620/169216837-e4c3354b-263f-4145-b53c-c285c91dfbbd.png">
</p>

The KMV Model can be seen as quite similar to the Black-Scholes Option Pricing model.

## **Analogy**

Equity holders of a firm can be seen as long a call option since they are in a limited liability agreement and can only enjoy the profits of a firm if the firm value is greater than its debt value. While the creditors of a firm can be seen as short a put option since they are only entitled to the principal and interest on their debt contribution. 
The model has 2 primary unknowns, which are, firm value and its volatility. We use the 3 approaches discussed in detail below to feed these unknowns into the model.

### **Method-1: Naive Estimation**

Method 1 is naive in the sense that firm value is approximated as the sum of face value of debt and market value of equity, while the volatility of firm value is approximated as weighted average of volatility of debt and equity. Plugging the values for all these variables in the equation, we get DD naive estimate in the range of 0 to 15 over the period of 50 years, apart from a few years around 1990 where the DD estimate has reached unreasonable levels. Below graph depicts the variation of quartiles of DD estimates over time.

<p align="center">
<img width="370" alt="image" src="https://user-images.githubusercontent.com/65303620/169217316-620e71d5-9548-4501-94c6-b7e4da6e3ab4.png">
</p>

A spike in the DD estimates has been observed which can be attributed to either an unreasonably high past year stock returns or stock return volatilities for a few of the securities.

### **Method 2: Direct Solving**

Method 2 leverages the Ito's Lemma to come up with the equations below for simultaneous solving of firm value, volatility of firm's value for each of the firms throughout the time period. \\\\ These estimated can then be plugged into the primary equation for Distance-to-Default and Probability-of-Default estimate.

<p align="center">
<img width="757" alt="image" src="https://user-images.githubusercontent.com/65303620/169217584-bde7aac8-b88f-4759-8a3f-02dae8956707.png">
</p>

The firm value, volatility of the firm’s value has been calculated using fsolve as the optimizer from the scipy package in pandas, for each year. Below graph depicts the variation of DD estimates using Method-2 over time.

<p align="center">
<img width="385" alt="image" src="https://user-images.githubusercontent.com/65303620/169217684-c2dc71af-1e8c-4a11-a504-55c35ca84b41.png">
</p>

As evident from the graph, the DD estimates seem to be highly different from Method-1and erroneous which can be primarily attributed to unrealistic estimates of V (firm’s value) and volatility of the firm’s value from using the scipy solver for few of the firm’s during 1980s due to non-convergence. As a result, many of the reasonable estimates in other years during the entire time horizon aren’t visible on the graph.

Thus, Method-2, though, a more complex and closer-to-reality approach seems to produce erroneous estimates during a couple of years in late 1970’s and 1980’s.

### **Method 3: Iterative Solving**

Method-3 is an iterative approach for computation of Distance-to-Default and Probability-of- Default, which is a refined and extended version of the Method- 2 computation of DD, PD. For this method, to reduce computational complexity, a random sample of 250 firms has been selected from the earlier sampled 1000 firms for Method-1 and Method-2 and leverages the daily data again to compute annualized volatility for each firm.
Below graph depicts the variation of DD over the period of 50 years calculated using Method-3 and seems to produce the most accurate estimates amongst all the 3 methods.

<p align="center">
<img width="456" alt="image" src="https://user-images.githubusercontent.com/65303620/169218066-84311aa0-f31c-42f1-8de4-c458def9e59c.png">
</p>

Below tabulated are the statistics of PD, DD measures using all the 3 methods:

<p align="center">
<img width="588" alt="image" src="https://user-images.githubusercontent.com/65303620/169218196-70e60842-bd28-4d7b-85c5-c2caa06483d0.png">
</p>

As evident from the above tables, statistics of the DD, PD estimates for Method-1 and Method-3 seem to be coherent, while the Method-2 estimates seem to be distorted due to non-convergence of firm value, vol. of firm’s value for a few of the data points.
Below is also evident from the correlation values observed between all the 3 methods.

<p align="center">
<img width="787" alt="image" src="https://user-images.githubusercontent.com/65303620/169218268-1e3b4fc2-7a83-456b-8edc-de7a989fa35b.png">
</p>

An interesting observation is the inverse correlation of DD, PD estimates across all the 3 methods which makes sense given a higher distance to default should imply lower default probability and a lower distance to default should imply higher default probability.

## **NBER Recession**

An interesting trend observed for DD, computed using Method-1 and Method-3 is a dip in the distance-to-default values prior to periods of economic recession.

**Method 1**

<p align="center">
<img width="381" alt="image" src="https://user-images.githubusercontent.com/65303620/169218536-8c4b966c-492b-446a-a9d7-b132488084bf.png">
</p>

**Method-3**

<p align="center">
<img width="385" alt="image" src="https://user-images.githubusercontent.com/65303620/169218607-d706942b-284c-4593-b5ef-d91760eb6ea0.png">
</p>

## **Fed-Fund Spread**

**Definition**:The difference between the average yield that a financial institution receives from loans—along with other interest-accruing activities—and the average rate it pays on deposits and borrowing.

<p align="center">
<img width="429" alt="image" src="https://user-images.githubusercontent.com/65303620/169218747-b80323d3-4150-49b3-b3f0-3d126a8db28b.png">
</p>

The Fed-Fund spread, quite evidently, seems to be inversely correlated with the variation of DD using Method-1, which seems to make sense since the distance-to-default for firms should lower during times of financial stress and uncertainty which is when spreads are expected to rise.

<p align="center">
<img width="438" alt="image" src="https://user-images.githubusercontent.com/65303620/169218805-6f37700c-544a-4f11-ba49-171de998d2f4.png">
</p>

## **Financial Stress Index**

**Method 1**

<p align="center">
<img width="434" alt="image" src="https://user-images.githubusercontent.com/65303620/169219174-2081ebab-8170-4249-8f89-c73783f5b92a.png">
</p>

**Method 3**

<p align="center">
<img width="459" alt="image" src="https://user-images.githubusercontent.com/65303620/169219252-a1082b3b-8f16-4883-a3a0-dc4638310d0a.png">
</p>

<p align="center">
<img width="488" alt="image" src="https://user-images.githubusercontent.com/65303620/169219300-2952c8ca-be30-4742-96c7-e8ec7804ad9a.png">
</p>

Though there are no clear patterns in the movement of the Financial Stress Index and Distance-to-Default over time for both the methods, but ideally, a dip in the distance-to- default estimates and a rise in the probability-of-default over time is expected with a rise in the financial stress index values.







