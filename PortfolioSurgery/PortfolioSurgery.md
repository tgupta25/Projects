# **Portfolio Surgery with Budget Constraints**

# **Objective**
This project aims to get rid of the negative payoff of a portfolio using a limited space of financial instruments with a given budget constraint.

# **Implementation Approach**
Given no budget constraints, the approach is a straight-forward one to regress the negative portion of the portfolio payoff with the payoff of given call and put options. However, to handle budget constraints, I perform a constrained multiple linear regression as below using the below call and puts.

<p align="center">
<img src="https://user-images.githubusercontent.com/65303620/171310650-c50062a7-49fc-4063-adcc-6e6bd252a9e8.png">
</p>

-To implement a constrained multiple linear regression in python, I modify the existing class, _constrained_linear_regression_ by introducing 2 new parameters A,B to adapt to working with constraints on linear combinations of regression coefficients.

-Given:
    -Portfolio payoff.
    -Put, call prices for different strikes.

**NOTE** : For the sake of implementation of this project, I only choose options with following strikes - 10,30,50,70,90.

-The portfolio's payoff before surgery looks like below:

<p align="center">
<img width="414" alt="Screen Shot 2022-05-31 at 6 47 05 PM" src="https://user-images.githubusercontent.com/65303620/171311681-37bc0643-6ed1-47ff-9f22-d4457d081800.png">
</p>


