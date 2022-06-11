%% Pricing option on option

%% Stock and other details

stock = 100;
sigma = 0.1;
rate = 0.03;
call_stk = 98;
call_maturity = 2;
calloncall_maturity = 1;

%% black scholes price of parent call
[parentcallprice notused] = blsprice(stock,call_stk,rate,...
    call_maturity,sigma);

calloncall_stk = parentcallprice;

%% simulate using gbm
dynamics = gbm(rate, sigma ,'StartState', stock);
steps = 360*2;
nTrials = 20000;
DeltaTime = 1/360;
s = simulate(dynamics, steps, 'nTrials',nTrials,'DeltaTime',DeltaTime);
s = squeeze(s); %%removes dimensions of length 1 with the same elements.

%% plot and see
plot(s);

%% Parent Price at 1-year point

parent_prices = blsprice(s(361,:), call_stk   ,rate,1,sigma);

%% call on call pricing

calloncall_price = ...
exp(-rate*calloncall_maturity)*...
    mean(max(parent_prices- calloncall_stk,0));




