% compare 2023_06_23_L2
clc
clear

WS = load('2023_06_23_L2.mat');
Y1 = WS.ppContainer(1);
Y2 = WS.ppContainer(2);
Y3 = WS.ppContainer(end);
YS = [Y1 Y3];

figure;
hold on;
lgnds=[];
for Y=YS
    plot(Y.acc*100);
    yline(Y.avg*100);
    lgnds=[lgnds;Y.Id;Y.Id+" average"];
end
hold off;
title("Accuracy vs Simulation Index for "+WS.numEpochs+" Epochs, "+WS.N_LAYERS+" Hidden Layers, Ideal (No Noise)");
xlabel("Simulation Index");
ylabel("Accuracy [%]");
legend(lgnds);
