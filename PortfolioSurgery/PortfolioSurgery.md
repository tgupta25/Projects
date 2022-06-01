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

-For this project, I take the surgery budget as $150,000 and perform a multiple linear regression with this budget constraint. The regression coefficients, which represent the number of options to be bought or sold, are as below.

<p align="center">
<img src="https://user-images.githubusercontent.com/65303620/171312983-f3d4507f-729e-49ce-9369-acf4e6311b45.png">
</p>

Below is a comparison of the portfolio payoffs before and after surgery.

<p align="center">
<img width="414" alt="image" src="https://user-images.githubusercontent.com/65303620/171328542-11f54d8a-8aec-475f-8f6c-4db546285bea.png">
</p>

Below is a closer look at the portfolio payoff after surgery.

<p align="center">
<img width="414" alt="image" src="https://user-images.githubusercontent.com/65303620/171328665-6bde6e68-b0eb-48fd-8607-47a4f3439f33.png">
</p>

#### **Conclusion**
To conclude, the cost of creating an opposite payoff to the negative portion of the portfolio payoff is np.dot(beta, prices) which is ~$33,417.

### **Future Work**

- A lot can be experimented with the learning rate of the constrained multiple linear regression to get better predictive power. This project has been implemented with a rate of 0.5.
- The results might be vastly different given the space of financial instruments for hedging.





