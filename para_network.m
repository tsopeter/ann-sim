% network
clc;
clear;

%% Testing parameters
run_stats = true;


%% network parameters

% get testing and training data
testingData  = create_imagedatastore('Images/t10k-labels-idx1-ubyte.gz','TestImagesPNG/');
trainingData = create_imagedatastore('Images/train-labels-idx1-ubyte.gz','TrainImagesPNG/');

X = linspace(0, 1, 256);
% make sure that functions are bounded between [0, 1] for X=[0, 1]

Y  = load('tf.mat').uu3;
XX  = linspace(0, 1, length(Y));
Y2 = Y - min(Y);
Y2 = Y2 ./ max(Y2);

PPS = {
    getpfit(X, @(X)(X), 6),...
    getpfit(X, @(X)(X.^2), 6),...
    getpfit(X, @(X)(X.^(1/3)), 6),...
    polyfit(XX, Y2, 6),...
};

PPSName = [
    "X",...
    "X^2",...
    "X^(1/3)",...
    "Normalized Fitted Transfer Function",...
];


%% Generate 'effects'
ef1=@(X)effect1f(X);
ef2=@(X)effect2f(X);
ef3=@(X)(max(max(X, [], [28 1])));

%% get the network parameters
learnRate     = 1e-5;
numEpochs     = 8;
miniBatchSize = 64;

ss = [size(imread(cell2mat(testingData.Files(1)))), 1];
kernel = exp(1i*zeros(ss));
lvalue=1e-6;


%% create the network layers
ppContainer(length(PPS))=Container();
for j=1:length(ppContainer)
    pp2=PPS{j};
    dd2=polyder(pp2);
    ppContainer(j).Id = PPSName(j);

    inputLayer     = imageInputLayer(ss, Name='input', Normalization='rescale-zero-one');
    kernelLayer    = CustomPhaseKernelLayer('kernel', kernel,1);
    protect1       = CustomNaNPreventionLayer('protect1',lvalue);
    protect2       = CustomNaNPreventionLayer('protect2',lvalue);
    amplitudeLayer = CustomPolarizationLayer('polar');
    Effect1        = functionLayer(ef1, Name='effect1', Formattable=true);
    max1           = functionLayer(ef3, Name='max1', Formattable=true);

    % DUT           = reluLayer(Name='dut');
    DUT           = CustomPolynomialNonLinearLayer('dut',pp2,dd2,ss,1,1);
       
    Effect2       = functionLayer(ef2, Name='effect2', Formattable=true);
    mult1         = CustomHProdLayer('hprod1');
    flatten       = fullyConnectedLayer(10, Name='flatten', WeightsInitializer='ones', BiasInitializer='ones');
    L2            = softmaxLayer(Name='L2');
    %L2            = sigmoidLayer("Name","L2");
    classifyy     = classificationLayer(Name='classify');
    
    layers = [
       inputLayer
       kernelLayer
       amplitudeLayer
       protect1
       protect2
       DUT
       flatten
       L2
       classifyy
    ];
    
    %% connect Layers
    lgraph = layerGraph();
    for i=1:length(layers)
        lgraph = addLayers(lgraph, layers(i));
    end
    
    lgraph = connectLayers(lgraph, 'input', 'kernel');
    lgraph = connectLayers(lgraph, 'kernel/out1', 'protect1');
    lgraph = connectLayers(lgraph, 'kernel/out2', 'protect2');
    
    lgraph = connectLayers(lgraph, 'protect1', 'polar/in1');
    lgraph = connectLayers(lgraph, 'protect2', 'polar/in2');
    
    lgraph = connectLayers(lgraph, 'polar', 'dut');
    lgraph = connectLayers(lgraph, 'dut', 'flatten');
    
    lgraph = connectLayers(lgraph, 'flatten', 'L2');
    lgraph = connectLayers(lgraph, 'L2', 'classify');
    
    %figure;
    %plot(lgraph);
    
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
    NRUNS=30;
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