clc; clear; close all;
%% ------------------------------------ Load data ------------------------------------
data = chickenpox_dataset;
data = [data{:}];
%% --------------------------
trec = numel(data);
trrec = 0.9 * trec;
NTST = floor(trrec);
datatrain = data(1:NTST + 1);
datatest = data(NTST +1 : end);
mu = mean(datatrain);
sig = std(datatrain);
datatrainstd = (datatrain - mu) / sig;
xtrain = datatrainstd(1:end-1);
ytrain = datatrainstd(2:end);
%%  Define LSTM Network Architecture
NOF=1;
NOR=1;
NHU=200;
layers=[
sequenceInputLayer(NOF, "Name", "ip")
lstmLayer(NHU, "Name", "lstm")
fullyConnectedLayer(NOR, "Name", "FC")
regressionLayer("Name", "RL")];
% lgraph=layerGraph(layers);
% plot (lgraph)
options = trainingOptions("adam",...
    "MaxEpochs", 250, ...
    "GradientThreshold", 1,...
    "InitialLearnRate", 0.005,...
    "LearnRateSchedule", "piecewise",...
    "LearnRateDropPeriod", 125, ...
    "LearnRateDropFactor", 0.2,...
    "Verbose", 0,...
    "Plots", "training-progress");
net = trainNetwork(xtrain , ytrain, layers, options);
datateststd = (datatest - mu) / sig;
xtest = datateststd(1:end-1);
ytest = datatest(2:end);
net = predictAndUpdateState(net, xtrain);
[net, ypred] = predictAndUpdateState(net, ytrain(end));
NTSTs = numel(datatest);
for i = 2:NTSTs
    [net, ypred(:, i)] = predictAndUpdateState(net, ytrain(:, i-1));
end
ypred = sig*ypred + mu;
rmse =sqrt(mean(ypred(2:end) - ytest).^2);
%% -------------------------------------- Plot ---------------------------------------
figure;
plot(data); hold on;
plot(NTST:NTST + NTSTs, [data(NTST) ypred], '.-')
xlabel("Months")
ylabel("Cases")
title("Monthly cases of chickenpox")