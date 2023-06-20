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
PPS = {
    getpfit(X, @(X)(X), 1),...
    getpfit(X, @(X)(X.^2), 2),...
    getpfit(X, @(X)(X.^(1/3)), 6),...
};

PPSName = [
    "X",...
    "X^2",...
    "X^(1/3)",...
];


%% Generate 'effects'
ef1=@(X)effect1f(X);
ef2=@(X)effect2f(X);
ef3=@(X)(max(max(X, [], [28 1])));

%% get the network parameters
learnRate     = 1e-5;
numEpochs     = 128;
miniBatchSize = 512;

ss = [size(imread(cell2mat(testingData.Files(1)))), 1];
kernel = abs(randn(ss));


%% create the network layers
ppContainer(length(PPS))=Container();
for j=1:length(ppContainer)
    pp=PPS(j);
    pp2=cell2mat(pp);
    dd2=polyder(pp2);
    ppContainer(j).Id = PPSName(j);

    inputLayer    = imageInputLayer(ss, Name='input', Normalization='rescale-zero-one');
    kernelLayer   = CustomAmplitudeKernelLayer('kernel', kernel);
    positiveLayer = CustomPositiveLayer('positive');
    Effect1       = functionLayer(ef1, Name='effect1', Formattable=true);
    max1          = functionLayer(ef3, Name='max1', Formattable=true);
    
    % DUT           = reluLayer(Name='dut');
    DUT           = CustomPolynomialNonLinearLayer('dut',pp2,dd2,ss,1,1);
           
    Effect2       = functionLayer(ef2, Name='effect2', Formattable=true);
    mult1         = CustomHProdLayer('hprod1');
    flatten       = fullyConnectedLayer(10, Name='flatten');
    %L2            = softmaxLayer(Name='L2');
    L2            = sigmoidLayer("Name","L2");
    classifyy     = classificationLayer(Name='classify');
    
    layers = [
       inputLayer
       kernelLayer
       positiveLayer
    
    
       max1
       Effect1
       DUT
       Effect2
       mult1
    
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
    lgraph = connectLayers(lgraph, 'kernel', 'positive');
    
    lgraph = connectLayers(lgraph, 'positive', 'effect1');
    lgraph = connectLayers(lgraph, 'positive', 'max1');
    lgraph = connectLayers(lgraph, 'effect1', 'dut');
    lgraph = connectLayers(lgraph, 'dut', 'effect2');
    lgraph = connectLayers(lgraph, 'effect2', 'hprod1/in1');
    lgraph = connectLayers(lgraph, 'max1', 'hprod1/in2');
    
    lgraph = connectLayers(lgraph, 'hprod1', 'flatten');
    lgraph = connectLayers(lgraph, 'flatten', 'L2');
    lgraph = connectLayers(lgraph, 'L2', 'classify');
    
    figure;
    plot(lgraph);
    
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
        Verbose=true,...
        ExecutionEnvironment='auto',...
        DispatchInBackground=false,...
        MiniBatchSize=miniBatchSize);
    end
    
    %% run network
    NRUNS=1;
    acc=zeros(NRUNS,1);
    if run_stats
        for i=1:NRUNS
            net = trainNetwork(trainingData, lgraph, options);
            YPred = classify(net,testingData);
            YValidation = testingData.Labels;
            
            accuracy = sum(YPred == YValidation)/numel(YValidation)
            acc(i)   = accuracy;
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