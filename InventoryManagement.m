clear all
clc
close all

% load data
productName = 'classicCar';
data = readtable(['./data/sales_data_sample_' productName '.xlsx']); 
n = size(data,1); % number of periods
D = data.QUANTITYORDERED;
X = NaN(n,1); % inventory on hand
A = NaN(n,1);
thresholds = [2000:1:4000];
lostPct = 0.05; % threshold for lost sale;
% Initiate result struct
out.thresholds = [];
out.sumInventory = [];
out.sumUnsat = [];
doAllData = 0;
doMPolicy = 0;
doNN = 0;
doNNConstrain = 1;
% Initiate first month inventory as 0
if doAllData == 1
X(1) = 0;
for ith = 1:numel(thresholds)
    threshold = thresholds(ith);
    output = inventorySimulation(D,threshold);
%    if sumUnsat <= .05 * sum(D) % lost sale is less than 5%
        out.thresholds = [out.thresholds, threshold];
        out.sumInventory = [out.sumInventory, output.sumInventory];
        out.sumUnsat = [out.sumUnsat, output.sumUnsat];
%    end
end

% Calculate optimal solution
mask = out.sumUnsat <= lostPct * sum(D);
minInv = min(out.sumInventory(mask));
minID = find(minInv == out.sumInventory);
minThr = out.thresholds(minID);
minUns = out.sumUnsat(minID);

saveName = ['./results/saleByMonth_' productName];
fig = AH_figure(1,1,saveName);
years = [2003,2004,2005];
for iyear = 1:numel(years)
    year = years(iyear);
    ymask = data.YEAR_ID == year;
    plot(data.MONTH_ID(ymask),data.QUANTITYORDERED(ymask))    
    hold on
end
title('Sale by month');
xlabel('Month');ylabel('Quantity sold');xlim([1,12]);
legend({'2003','2004','2005'},'location','northwest');
saveas(fig, [saveName '.fig']);
saveas(fig, [saveName '.png']);

%%
saveName = ['minInventory_' productName'];
fig = AH_figure(1.5,1.5,saveName);
subplot(2,1,1)
plot(out.thresholds, out.sumInventory);
ylabel('Total inventory (to minimize)');
xlabel('Policy threshold');
title({['Optimal policy threshold = ' num2str(minThr)];['Minimal total inventory = ' num2str(minInv)]});
vline(minThr); hline(minInv);

subplot(2,1,2)
plot(out.thresholds, out.sumUnsat);
hline(unsThresh, 'r-'); hline(lostPct*sum(D)); vline(minThr);
title(['Total unsatisfied demand = ' num2str(minUns)]);
ylabel({'Total unsatisfied demand';['(<=' num2str(lostPct) '*total demand = ' num2str(.05*sum(D)) ')']});
xlabel('Policy threshold');

saveas(fig, [saveName '.fig']);
saveas(fig, [saveName '.png']);
end

%% m-policy
if doMPolicy == 1
% Initiate first month inventory as 0
thresholds = [1000:1:5000];
lostPct = 0.05; % threshold for lost sale;
mRange = [2:15];
for iM = 1:numel(mRange) % how many previous time periods are used
    m = mRange(iM);
    mTrains = n-m-1;
    mX{iM}(1) = 0;
    mX{iM} = NaN(m+1,mTrains); % inventory on hand
    mA{iM} = NaN(m+1,mTrains); % order size
    mD{iM} = NaN(m+1,mTrains); % sales size
    
    for iTrain = 1:mTrains
        mD{iM}(:,iTrain) = D(iTrain : iTrain+m); % 1:m for each bin, last row is testing data
    end
    
    % Initiate result struct
    mout.thresholds{iM} = [];
    mout.avgInventory{iM} = [];
    mout.avgUnsat{iM} = [];
    mout.avgD{iM} = [];
    
    for ith = 1:numel(thresholds)
        threshold = thresholds(ith);
        for iTrain = 1:mTrains % simulation starts
            trainD = mD{iM}(:,iTrain);
            output = inventorySimulation(trainD,threshold);
            testInventory(iTrain) = output.testInventory;
            testUnsat(iTrain) = output.testUnsat;
        end
        mout.thresholds{iM} = [mout.thresholds{iM}, threshold];
        mout.avgInventory{iM} = [mout.avgInventory{iM}, nanmean(testInventory)];
        mout.avgUnsat{iM} = [mout.avgUnsat{iM}, nanmean(testUnsat)];
        mout.avgD{iM} = nanmean(mD{iM}(m+1,:)); % last row is test data
        clear testInventory testUnsat
    end
    
    % Calculate optimal solution
    unsThresh = lostPct * mout.avgD{iM};
    mask = mout.avgUnsat{iM} <= unsThresh;
    minInv = min(mout.avgInventory{iM}(mask));
    minID = find(minInv == mout.avgInventory{iM});
    minThr = mout.thresholds{iM}(minID);
    minUns = mout.avgUnsat{iM}(minID);
    
    saveName = ['minInventory_' productName '_m=' num2str(m)];
    fig = AH_figure(1.5,1.5,saveName);
    subplot(2,1,1)
    plot(mout.thresholds{iM}, mout.avgInventory{iM});
    ylabel('Average inventory (to minimize)');
    xlabel('Policy threshold');
    title({['Optimal policy threshold = ' num2str(minThr)];['Minimal average inventory = ' num2str(minInv)]});
    vline(minThr); hline(minInv);

    subplot(2,1,2)
    plot(mout.thresholds{iM}, mout.avgUnsat{iM});
    hline(unsThresh, 'r-');hline(minUns); vline(minThr);
    title(['Average unsatisfied demand = ' num2str(minUns)]);
    ylabel({'Average unsatisfied demand';['(<=' num2str(lostPct) '*avg demand = ' num2str(unsThresh) ')']});
    xlabel('Policy threshold');

    save([saveName],'m','mout');
    saveas(fig, [saveName '.fig']);
    saveas(fig, [saveName '.png']);
    
    % Load into hyper-parameter matrix
    best.threshold(iM) = minThr;
    best.avgInventory(iM) = minInv;
    best.avgUnsat(iM) = minUns;
end

% calculate best policy
minInv = min(best.avgInventory);
minID = find(minInv == best.avgInventory);
minThr = best.threshold(minID);
minUns = best.avgUnsat(minID);
minM = mRange(minID);
xLim = [mRange(1),mRange(end)];

saveName = ['minInventory_' productName '_m=' num2str(mRange(1)) '~' num2str(mRange(end))];
fig = AH_figure(1.5,1.5,saveName);
subplot(2,1,1)
plot(mRange, best.avgInventory);
ylabel('Average inventory (to minimize)');
xlabel('Policy m selection');
title({['Optimal policy: m = ' num2str(minM) ', threshold = ' num2str(minThr)];['Minimal average inventory = ' num2str(minInv)]});
vline(minM); hline(minInv);
xlim(xLim);

subplot(2,1,2)
plot(mRange, best.avgUnsat);
hline(minUns); vline(minM); % don't need threshold
title(['Average unsatisfied demand = ' num2str(minUns)]);
ylabel({'Average unsatisfied demand';['(<=' num2str(lostPct) '*avg demand)']});
xlabel('Policy m selection');
xlim(xLim);

save([saveName],'mRange','best');
saveas(fig, [saveName '.fig']);
saveas(fig, [saveName '.png']);

end

%% Neural Net
if doNN == 1
mRange = [2:15];
for iM = 1:numel(mRange) % how many previous time periods are used
    m = mRange(iM);
    %m = 11; % pick previous result
    mTrains = n-m-1;
    mX{iM}(1) = 0;
    mX{iM} = NaN(m+1,mTrains); % inventory on hand
    mA{iM} = NaN(m+1,mTrains); % order size
    mD{iM} = NaN(m+1,mTrains); % sales size
    
    for iTrain = 1:mTrains
        mD{iM}(:,iTrain) = D(iTrain : iTrain+m); % 1:m for each bin, last row is testing data
    end
    
    % Initiate result struct
    mout.thresholds{iM} = [];
    mout.avgInventory{iM} = [];
    mout.avgUnsat{iM} = [];
    mout.avgD{iM} = [];
    
    for iSplit = 1:100 % split 10 times to calculate testing acc
        cv = cvpartition(size(mD{iM},2),'HoldOut',0.2);
        idx = cv.test;
        % seperate training and test data
        trainX = mD{iM}(1:end-1,~idx);
        trainY = mD{iM}(end,~idx);
        testX = mD{iM}(1:end-1,idx);
        testY = mD{iM}(end,idx);
        
        % create NN
        netconf = [3];
        net = feedforwardnet(netconf); % 1 hidden neuron with 3 neurons
        net = train(net,trainX,trainY);
        ypred = net(testX);
        
        mout.threshold{iM}(iSplit,:) = ypred;
        mout.avgInventory{iM}(iSplit) = mean(ypred);
        mout.avgUnsat{iM}(iSplit) = mean(max(0,testY-ypred));
        mout.avgD{iM}(iSplit) = mean(testY);
        mout.loss{iM}(iSplit) = mean((ypred-testY).^2);
        
    end
    mout.lossMn(iM) = nanmean(mout.loss{iM});
    mout.lossStd(iM) = nanstd(mout.loss{iM});
    mout.avgUnsatPercMn(iM) = nanmean(mout.avgUnsat{iM}./mout.avgD{iM});
    mout.avgUnsatPercStd(iM) = nanstd(mout.avgUnsat{iM}./mout.avgD{iM});
    mout.avgInvMn(iM) = nanmean(mout.avgInventory{iM});
    mout.avgInvStd(iM) = nanstd(mout.avgInventory{iM});

%     % For a specific m, each iteration
%     fig = AH_figure(1,1,'');
%     plot([1:10],mout.avgInventory{iM});
%     hold on;
%     plot([1:10],mout.avgD{iM}); ylabel('Quantity per month')
%     xlabel('Iteration'); legend({'Average inventory','Average demand'});
    
    
%     % Calculate optimal solution
%     unsThresh = lostPct * mout.avgD{iM};
%     mask = mout.avgUnsat{iM} <= unsThresh;
%     minInv = min(mout.avgInventory{iM}(mask));
%     minID = find(minInv == mout.avgInventory{iM});
%     minThr = mout.thresholds{iM}(minID);
%     minUns = mout.avgUnsat{iM}(minID);
end

best.Loss = min(mout.lossMn);
best.ID = find(best.Loss == mout.lossMn);
best.m = mRange(best.ID);
best.UnsPerc = mout.avgUnsatPercMn(best.ID);
best.Inv = mout.avgInvMn(best.ID);
xLim = [mRange(1),mRange(end)];

saveName = ['NN_prediction_' productName '_m=' num2str(mRange(1)) '~' num2str(mRange(end))];
save([saveName],'mRange','mout','best');

fig = AH_figure(2,2,saveName);
subplot(3,1,1)
shadedErrorBar(mRange,mout.lossMn,mout.lossStd,{'k-o','markerfacecolor','k'});
ylabel('Mean square loss (to minimize)');
xlabel('Policy m selection');
title({['Optimal policy: m = ' num2str(best.m)];['Minimal square loss = ' num2str(round(best.Loss)) '; Minimal error = ' num2str(round(sqrt(best.Loss)))]});
vline(best.m); hline(best.Loss);
xlim(xLim);

subplot(3,1,2)
shadedErrorBar(mRange,mout.avgInvMn,mout.avgInvStd,{'k-o','markerfacecolor','k'});
ylabel('Average inventory');
xlabel('Policy m selection');
title({['Average inventory = ' num2str(round(best.Inv))]});
vline(best.m); hline(best.Inv);
xlim(xLim);

subplot(3,1,3)
shadedErrorBar(mRange,mout.avgUnsatPercMn,mout.avgUnsatPercStd,{'k-o','markerfacecolor','k'});
hline(best.UnsPerc); vline(best.m); % don't need threshold
title(['Average proportion unsatisfied demand = ' num2str(best.UnsPerc*100) '%']);
ylabel({'Average unsatisfied demand%'});
xlabel('Policy m selection');
xlim(xLim);

saveas(fig, [saveName '.fig']);
saveas(fig, [saveName '.png']);
end

%% Neural Net with constrain, which is added to the mse function, gradient descent function is not modified, hope there is no problem
if doNNConstrain == 1
 
% %%%BEGIN CODE%%%
% % Thanks to Amaria Zidouk, Technical Support Team, MathWorks for this...
% % Make a copy of mse function in your current working directory
% % (Note: mae, sse, sae could also be used as starting points)
% copyfile(['C:\Program Files\MATLAB\R2020a\toolbox\nnet\nnet\nnperformance\mse.m'], 'mymse.m');
% % load a simple example input and target dataset
% [x,t] = simplefit_dataset; 
% % Create an example fitting ANN with 10 hidden units and using the scaled 
% % conjugate gradients training function
% net = fitnet(10,'trainscg'); 
% % Change the performance function (fitness function) to our local copy
% % Note!! The name of the function inside mymse.m *MUST* be kept as 'mse' 
% % as in:
% % function [out1,out2] = mse(varargin)
% % This ensures our user defined function is overloaded in place of the
% % standard function 'mse' and that the error checking inside NN toolbox 
% % does not throw up an error!
% net.performFcn = 'mymse'; 
% % Prove that our new (so far identical) mymse.m function can be used
% % correctly and works.
% net = train(net,x,t); 
% % You are now free to modify mymse.m to suit your new performance/fitness
% % function requirements 
% %%%END CODE%%%
    
mRange = [2:15];
pDemand = 10000;
for iM = 1:numel(mRange) % how many previous time periods are used
    m = mRange(iM);
    %m = 11; % pick previous result
    mTrains = n-m-1;
    mX{iM}(1) = 0;
    mX{iM} = NaN(m+1,mTrains); % inventory on hand
    mA{iM} = NaN(m+1,mTrains); % order size
    mD{iM} = NaN(m+1,mTrains); % sales size
    
    for iTrain = 1:mTrains
        mD{iM}(:,iTrain) = D(iTrain : iTrain+m); % 1:m for each bin, last row is testing data
    end
    
    % Initiate result struct
    mout.thresholds{iM} = [];
    mout.avgInventory{iM} = [];
    mout.avgUnsat{iM} = [];
    mout.avgD{iM} = [];
    
    for iSplit = 1:100 % split 10 times to calculate testing acc
        cv = cvpartition(size(mD{iM},2),'HoldOut',0.2);
        idx = cv.test;
        % seperate training and test data
        trainX = mD{iM}(1:end-1,~idx);
        trainY = mD{iM}(end,~idx);
        testX = mD{iM}(1:end-1,idx);
        testY = mD{iM}(end,idx);
        
        % create NN
        netconf = [3];
        net = fitnet(netconf,'trainscg'); % 1 hidden neuron with 3 neurons
        net.layers{1}.transferFcn = 'poslin'; % a type of ReLu
        net.performFcn = 'mymse'; % Use customized performance function
        % make sure folder +mymse exist in matlab folder
        net = train(net,trainX,trainY);
        ypred = net(testX);
        
        mout.threshold{iM}(iSplit,:) = ypred;
        mout.avgInventory{iM}(iSplit) = mean(ypred);
        mout.avgUnsat{iM}(iSplit) = mean(max(0,testY-ypred));
        mout.avgD{iM}(iSplit) = mean(testY);
        mout.avgUnsatPer{iM}(iSplit) = mean(max(testY-ypred,0)./testY); % Get % for each test sample
        %mout.avgUnsatPer{iM}(iSplit) = mout.avgUnsat{iM}(iSplit) / mout.avgD{iM}(iSplit); % calculate avg lost demand
        mout.loss{iM}(iSplit) = mean((ypred-testY).^2);
        mout.lossTotal{iM}(iSplit) = mean((ypred-testY).^2 + pDemand * max(testY-ypred,0));
        
    end
    mout.lossMn(iM) = nanmean(mout.loss{iM});
    mout.lossStd(iM) = nanstd(mout.loss{iM});
    mout.lossTotalMn(iM) = nanmean(mout.lossTotal{iM});
    mout.lossTotalStd(iM) = nanstd(mout.lossTotal{iM});
    mout.avgUnsatPercMn(iM) = nanmean(mout.avgUnsatPer{iM});
    mout.avgUnsatPercStd(iM) = nanstd(mout.avgUnsatPer{iM});
    mout.avgInvMn(iM) = nanmean(mout.avgInventory{iM});
    mout.avgInvStd(iM) = nanstd(mout.avgInventory{iM});

%     % For a specific m, each iteration
%     fig = AH_figure(1,1,'');
%     plot([1:10],mout.avgInventory{iM});
%     hold on;
%     plot([1:10],mout.avgD{iM}); ylabel('Quantity per month')
%     xlabel('Iteration'); legend({'Average inventory','Average demand'});
    
    
%     % Calculate optimal solution
%     unsThresh = lostPct * mout.avgD{iM};
%     mask = mout.avgUnsat{iM} <= unsThresh;
%     minInv = min(mout.avgInventory{iM}(mask));
%     minID = find(minInv == mout.avgInventory{iM});
%     minThr = mout.thresholds{iM}(minID);
%     minUns = mout.avgUnsat{iM}(minID);
end
mask = mout.avgUnsatPercMn <= lostPct;
best.Loss = min(mout.lossMn(mask));
best.LossTotal = min(mout.lossTotalMn(mask));
best.ID = find(best.Loss == mout.lossMn);
best.m = mRange(best.ID);
best.UnsPerc = mout.avgUnsatPercMn(best.ID);

best.Inv = mout.avgInvMn(best.ID);
xLim = [mRange(1),mRange(end)];

saveName = ['NNConstrain_prediction_' productName '_m=' num2str(mRange(1)) '~' num2str(mRange(end)) '_p=' num2str(pDemand)];
save([saveName],'mRange','mout','best');

fig = AH_figure(3,2,saveName);
subplot(4,1,1)
shadedErrorBar(mRange,mout.lossMn,mout.lossStd,{'k-o','markerfacecolor','k'});
ylabel({'Loss = Mean square loss'; '+ lost-demand cost'});
xlabel('Policy m selection');
title({['Optimal policy: m = ' num2str(best.m)];['Minimal total loss = ' num2str(round(best.LossTotal))]});
vline(best.m); hline(best.Loss);
xlim(xLim);

subplot(4,1,2)
shadedErrorBar(mRange,mout.lossMn,mout.lossStd,{'k-o','markerfacecolor','k'});
ylabel('Mean square loss');
xlabel('Policy m selection');
title({['Mean square loss = ' num2str(round(best.Loss)) '; Minimal mean error = ' num2str(round(sqrt(best.Loss)))]});
vline(best.m); hline(best.Loss);
xlim(xLim);

subplot(4,1,3)
shadedErrorBar(mRange,mout.avgInvMn,mout.avgInvStd,{'k-o','markerfacecolor','k'});
ylabel('Average inventory');
xlabel('Policy m selection');
title({['Average inventory = ' num2str(round(best.Inv))]});
vline(best.m); hline(best.Inv);
xlim(xLim);

subplot(4,1,4)
shadedErrorBar(mRange,mout.avgUnsatPercMn,mout.avgUnsatPercStd,{'k-o','markerfacecolor','k'});
hline(best.UnsPerc); hline(lostPct,'r-');vline(best.m); % don't need threshold
title(['Average proportion unsatisfied demand = ' num2str(best.UnsPerc*100) '%']);
ylabel({'Average unsatisfied demand%'});
xlabel('Policy m selection');
xlim(xLim);

saveas(fig, [saveName '.fig']);
saveas(fig, [saveName '.png']);
end
