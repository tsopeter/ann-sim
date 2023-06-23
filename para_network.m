% network
clc;
clear;

%% Testing parameters
run_stats = false;
run_select = 1;

%% network parameters

% get testing and training data
testingData  = create_imagedatastore('Images/t10k-labels-idx1-ubyte.gz','TestImagesPNG/');
trainingData = create_imagedatastore('Images/train-labels-idx1-ubyte.gz','TrainImagesPNG/');

M = 3;
X = linspace(0, M, 256);

Y  = load('tf.mat').uu3;
XX  = linspace(0, M, length(Y));
Y2 = Y - min(Y);
Y2 = Y2 ./ max(Y2);

PPS = {
    getpfit(X, @(X)((1/M).*X), 6),...
    getpfit(X, @(X)(((1/M).*X).^2), 6),...
    getpfit(X, @(X)(((1/M).*X).^(1/2)), 11),...
    polyfit(XX, Y2, 4),...
};

PPSName = [
    "X",...
    "X^2",...
    "X^{1/2}",...
    "Normalized Fitted Transfer Function",...
];

if run_stats == false
    PPS = PPS(run_select);
    PPSName = PPSName(run_select);
end

%% plot transfer functions
figure;
hold on;
lgnds=[];
for i=1:length(PPS)
    pp=PPS{i};
   plot(X, polyval(pp, X));
   lgnds=[lgnds;PPSName(i)];
end
hold off;
title("Transfer Functions under Test");
xlabel("Input");
ylabel("Output");
legend(lgnds);

figure;
hold on;
lgnds=[];
for i=1:length(PPS)
    pp=PPS{i};
   plot(X, polyval(polyder(pp), X));
   lgnds=[lgnds;PPSName(i)];
end
hold off;
title("Derivative of Transfer Functions under Test");
xlabel("Input");
ylabel("Output");
legend(lgnds);


%% Generate 'effects'
ef1=@(X)effect1f(X);
ef2=@(X)effect2f(X);
ef3=@(X)(max(max(X, [], [28 1])));

%% get the network parameters
learnRate     = 1e-5;
numEpochs     = 8;
miniBatchSize = 64;

ss = [size(imread(cell2mat(testingData.Files(1)))), 1];
kernel = abs(randn(ss));
lvalue=1e-8;
c = 0.5;


%% create the network layers
ppContainer(length(PPS))=Container();
for j=1:length(ppContainer)
    pp2=PPS{j};
    dd2=polyder(pp2);
    ppContainer(j).Id = PPSName(j);

    inputLayer     = imageInputLayer(ss, Name='input', Normalization='rescale-zero-one');
    kernelLayer    = CustomAmplitudeKernelLayer('kernel', kernel);
    protect1       = CustomNaNPreventionLayer('protect1', lvalue);
    positiveLayer  = CustomPositiveLayer('post1');
    add1           = CustomConstantAddLayer('add1', c);

    % DUT           = reluLayer(Name='dut');
    A1            = CustomAnalyzerLayer('a1');
    DUT           = CustomPolynomialNonLinearLayer('dut',pp2,dd2,ss,1,1);

    % second linear layer
    L1            = fullyConnectedLayer(128, Name='L1', WeightsInitializer='glorot', BiasInitializer='narrow-normal');

    % force weights to be positive
    positiveLayer2 = CustomPositiveLayer('post2');

    A2            = CustomAnalyzerLayer('a2');

    DUT2          = CustomPolynomialNonLinearLayer('dut2', pp2, dd2, ss, 1, 1);
       
    flatten       = fullyConnectedLayer(10, Name='flatten', WeightsInitializer='glorot', BiasInitializer='narrow-normal');
    L2            = softmaxLayer(Name='L2');
    %L2            = sigmoidLayer("Name","L2");
    classifyy     = classificationLayer(Name='classify');
    
    layers = [
       inputLayer       % simulates optical side
       kernelLayer
       protect1
       positiveLayer
       add1
       
       %A1
       DUT
       L1
       positiveLayer2
       A2
       DUT2


       flatten          % digital side
       L2
       classifyy
    ];
    
    %% connect Layers
    lgraph = layerGraph();
    for i=1:length(layers)
        lgraph = addLayers(lgraph, layers(i));
    end
    
    lgraph = connectLayers(lgraph, 'input', 'kernel');
    lgraph = connectLayers(lgraph, 'kernel', 'protect1');
    lgraph = connectLayers(lgraph, 'protect1', 'post1');

    lgraph = connectLayers(lgraph, 'post1', 'add1');
    lgraph = connectLayers(lgraph, 'add1', 'dut');
    lgraph = connectLayers(lgraph, 'dut', 'L1');
    lgraph = connectLayers(lgraph, 'L1', 'a2');
    lgraph = connectLayers(lgraph, 'a2', 'dut2');
    lgraph = connectLayers(lgraph, 'dut2', 'flatten');

    lgraph = connectLayers(lgraph, 'flatten', 'L2');
    lgraph = connectLayers(lgraph, 'L2', 'classify');
    
    if run_stats == false
        figure;
        plot(lgraph);
    end
    
    %% give options to the network
    
    if run_stats == false
    options = trainingOptions('adam',...
        InitialLearnRate=learnRate,...
        MaxEpochs=numEpochs,...
        Shuffle='every-epoch',...
        ValidationData=testingData,...
        ValidationFrequency=512,...
        Verbose=true,...
        Plots='training-progress',...
        ExecutionEnvironment='auto',...
        DispatchInBackground=false,...
        MiniBatchSize=miniBatchSize);
    else
        options = trainingOptions('adam',...
        InitialLearnRate=learnRate,...
        MaxEpochs=numEpochs,...
        Shuffle='every-epoch',...
        ValidationData=testingData,...
        ValidationFrequency=512,...
        Verbose=false,...
        ExecutionEnvironment='cpu',...
        DispatchInBackground=false,...
        MiniBatchSize=miniBatchSize);
    end
    
    %% run network
    NRUNS=8;
    acc=zeros(NRUNS,1);
    disp("Running "+j+" cell");
    if run_stats
        parfor i=1:NRUNS
            disp("Run "+i+" scheduled");
            net = trainNetwork(trainingData, lgraph, options);
            YPred = classify(net,testingData);
            YValidation = testingData.Labels;
            
            accuracy = sum(YPred == YValidation)/numel(YValidation)
            acc(i)   = accuracy;
            disp("Run "+i+" finished");
        end
    
    %% statistics
        avg_accuracy = mean(acc)
        sdv_accuracy = std(acc)
        var_accuracy = sdv_accuracy.^2
    
    
        ppContainer(j).acc = acc;
        ppContainer(j).avg = avg_accuracy;
        ppContainer(j).sdv = sdv_accuracy;
        ppContainer(j).var = var_accuracy;
    else
        net = trainNetwork(trainingData, lgraph, options);
        YPred = classify(net,testingData);
        YValidation = testingData.Labels;
        
        accuracy = sum(YPred == YValidation)/numel(YValidation)
    end
end

if run_stats
    %% plot results
    figure;
    lgnds=[];
    hold on;
        for cc=ppContainer
            plot(cc.acc.*100);
            lgnds=[lgnds;cc.Id];
        end
    hold off;
    title("Accuracy vs Simulation Count");
    xlabel("Simulation Count");
    ylabel("Accuracy");
    legend(lgnds);
end