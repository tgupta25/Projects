%% Rebate barrier options

%% Stock and other details
stock = 100;
sigma = 0.1;
rate = 0.03;

%% simulate using gbm
dynamics = gbm(rate, sigma ,'StartState', stock);
steps = 360*1;
nTrials = 20000;
DeltaTime = 1/360;
s = simulate(dynamics, steps, 'nTrials',nTrials,'DeltaTime',DeltaTime);
s = squeeze(s);
%%
%% cash or nothing payoffs
payoffs = 1000* (max(s)>120);
timing = nan(nTrials,1)';
aux = [(s > 120) ;ones(1,nTrials)]; %%appends ones(1,nTrials) to the matrix.
%% payoff timing!
s
for i = 1:nTrials
     timing(i)=min(find(aux(:,i)==1));
end

%% find price of rebate barrier

price_rebate = mean(exp(-rate*timing/360).*...
    payoffs)

%% compare to normal cash or nothing barrier

price_normal = mean(exp(-rate*1)*...
    payoffs)
