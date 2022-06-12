
%% stock data
stock1 = 100;
return1 = 0.03;
sigma1 = 0.20;

stock2 = 100;
return2 = 0.02;
sigma2 = 0.30;

%% create return and sigma
%% matrices
Return = diag([return1 return2]);
Sigma = diag([sigma1 sigma2]);

%% 2-dimentional gbm
correlation = [1 0.2; 0.2 1];
stocks = gbm(Return, Sigma,...
    'StartState' ,[100; 100],...
'correlation', correlation);

%% simulations!
DeltaTime = 1/360;
nobs = 360;
nTrials = 20000;
ss = simulate(stocks,nobs, ...
    'DeltaTime', DeltaTime,...
    'nTrials', nTrials);

%% extract stocks
s1 = squeeze(ss(:,1,:));
s2 = squeeze(ss(:,2,:));

%% see corresponding plots
tt = [s1(:,55) s2(:,55)];
    plot(tt);

%% price rainbow option
rate = 0.03;
rainbow_payoff =...
    max(s1(end,:), s2(end,:));
rainbow_price =...
    mean(rainbow_payoff*...
    exp(-rate*nobs*DeltaTime));
%%
%% Price exchange option
exchange_payoff =...
    max(s1(end,:)- s2(end,:),0);
exchange_price =...
    mean(exchange_payoff*...
    exp(-rate*nobs*DeltaTime));
