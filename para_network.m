% network
clc;
clear;

%% Testing parameters
run_stats = true;
run_select = 6;
N_RUNS    = 2;

%% network parameters

% get testing and training data
testingData  = create_imagedatastore('Images/t10k-labels-idx1-ubyte.gz','TestImagesPNG/');
trainingData = create_imagedatastore('Images/train-labels-idx1-ubyte.gz','TrainImagesPNG/');

M = 1;
X = linspace(0, M, 256);

Y  = load('tf.mat').uu3;
XX  = linspace(0, M, length(Y));
Y2 = Y - min(Y);
Y2 = Y2 ./ max(Y2);

CV=load('CURVEFIT/cvfit.mat');

NCV = CV.fn(CV.W-[CV.W(1) 0 0 0], 1);
SCV = CV.fn(CV.W, 0);
SSCV = CV.fn(CV.W, 1)-SCV;
PPS = {
    getpfit(X, @(X)((1/M).*X), 1),...
    getpfit(X, @(X)(((1/M).*X).^2), 6),...
    polyfit(XX, Y-min(Y), 4),...
    polyfit(XX, Y2, 4),...
    @(X)(CV.fn(CV.W, X)),...
    @(X)(CV.fn(CV.W-[CV.W(1) 0 0 0], X)),...
    @(X)(CV.fn(CV.W, X)-SCV)/SSCV,...
};

PPSName = [
    "X",...
    "X^2",...
    "Fitted Transfer Function",...
    "Normalized Fitted Transfer Function",...
    "Fitted Nonlinear Curve",...
    "Shifted Fitted Nonlinear Curve",...
    "Normalized Scaled Fitted Nonlinear Curve",...
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
    if ~isa(pp, 'function_handle')
        plot(X, polyval(pp, X));
    else
        plot(X, pp(X));
    end
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
    if ~isa(pp, 'function_handle')
        plot(X, polyval(polyder(pp), X));
    else
        plot(X(1:end-1), diff(pp(X)).*length(X));
    end
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
learnRate     = 6e-5;
numEpochs     = 12;
miniBatchSize = 96;

ss = [size(imread(cell2mat(testingData.Files(1)))), 1];
kernel = abs(randn(ss));
lvalue=1e-10;
c1 = 0.5;
c2 = 0.0;


%% create the network layers
ppContainer(length(PPS))=Container();
for j=1:length(ppContainer)
    pp2=PPS{j};
    if ~isa(pp2, 'function_handle')
        dd2=polyder(pp2);
    end
    ppContainer(j).Id = PPSName(j);

    inputLayer     = imageInputLayer(ss, Name='input', Normalization='rescale-zero-one');
    kernelLayer    = CustomAmplitudeKernelLayer('kernel', kernel);
    protect1       = CustomNaNPreventionLayer('protect1', lvalue);
    positiveLayer  = CustomPositiveLayer('post1');
    add1           = CustomConstantAddLayer('add1', c1);
    sat1           = CustomSaturationLayer('sat1', M);

    % DUT           = reluLayer(Name='dut');
    A1            = CustomAnalyzerLayer('a1', M);

    if ~isa(pp2, 'function_handle')
        DUT = CustomPolynomialNonLinearLayer('dut',pp2,dd2,ss,1,1);
    else
        DUT = functionLayer(pp2, Name='dut');
    end

    % second linear layer
    L1            = CustomAmplitudeKernelLayer('L1', kernel);

    % force weights to be positive
    positiveLayer2 = CustomPositiveLayer('post2');
    add2          = CustomConstantAddLayer('add2', c2);
    sat2          = CustomSaturationLayer('sat2', M);

    A2            = CustomAnalyzerLayer('a2', M);

    if ~isa(pp2, 'function_handle')
        DUT2 = CustomPolynomialNonLinearLayer('dut2', pp2, dd2, ss, 1, 1);
    else
        DUT2 = functionLayer(pp2, Name='dut2');
    end
       
    flatten       = fullyConnectedLayer(10, Name='flatten', WeightsInitializer='glorot', BiasInitializer='narrow-normal');
    %L2            = softmaxLayer(Name='L2');
    L2            = sigmoidLayer("Name","L2");
    classifyy     = classificationLayer(Name='classify');
    
    layers = [
       inputLayer       % simulates optical side
       kernelLayer
       protect1
       positiveLayer
       add1
       sat1
       
       %A1
       DUT
       L1
       positiveLayer2
       A2
       sat2
       add2
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
    lgraph = connectLayers(lgraph, 'add1', 'sat1');
    lgraph = connectLayers(lgraph, 'sat1', 'dut');
    lgraph = connectLayers(lgraph, 'dut', 'L1');
    lgraph = connectLayers(lgraph, 'L1', 'post2');
    lgraph = connectLayers(lgraph, 'post2', 'add2');
    lgraph = connectLayers(lgraph, 'add2', 'sat2');
    lgraph = connectLayers(lgraph, 'sat2', 'a2');
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
    NRUNS=N_RUNS;
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