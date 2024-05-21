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


%% -------------------------------------- Plot ---------------------------------------
figure;
plot(data)
xlabel("Months")
ylabel("Cases")
title("Monthly cases of chickenpox")