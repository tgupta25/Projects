%% stock details
stock = 100;
stock_sigma = 0.25;
rate = 0.03;
level = 0.30;
speed = 1;
volvol = 0.1;

%%
heston_model = heston(rate,speed,level,volvol,...
    'correlation',1,...
    'StartState', [stock;stock_sigma]);
%%
%% simulate!
nobs =720;
delta = 1/360;
nTrials = 7000;
ss = simulate(heston_model, nobs,...
    'DeltaTime', delta,...
    'nTrials', nTrials);

%% now price call with heston
call_mat =2;
call_strikes = [80:5:120];
paths = squeeze(ss(:,1,:));
call_payoffs = nan(1,length(call_strikes));
for i = 1:length(call_strikes)
    for j = 1: nTrials
    call_payoffs(j,i) = max(paths(end,j)- call_strikes(i),0);
    call_prices = mean(exp(-rate*call_mat)*...
    call_payoffs);
end
end
%% Now imply volatility using BS
imp_vol = blsimpv(stock,...
    call_strikes,rate, call_mat,...
    call_prices,'Yield',0,'Limit', 0.5,'Class', {'Call'});
plot(imp_vol);