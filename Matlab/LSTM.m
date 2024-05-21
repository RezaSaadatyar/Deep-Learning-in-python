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

%% -------------------------------------- Plot ---------------------------------------
figure;
plot(data)
xlabel("Months")
ylabel("Cases")
title("Monthly cases of chickenpox")