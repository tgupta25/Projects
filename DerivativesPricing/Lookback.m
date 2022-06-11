%% Lookback Call Pricing

%% Stock and other details

stock = 100;
sigma = 0.1;
rate = 0.03;
%call_stk = 98;
lookback_call_maturity = 2;

%% simulate using gbm
dynamics = gbm(rate, sigma ,'StartState', stock);
steps = 360*2;
nTrials = 20000;
DeltaTime = 1/360;
s = simulate(dynamics, steps, 'nTrials',nTrials,'DeltaTime',DeltaTime);
s = squeeze(s);

%% Lookback Pricing

lookback_payoffs = s(end,:) - min(s);

lookback_price = mean(exp(-rate*2)*...
    lookback_payoffs)



