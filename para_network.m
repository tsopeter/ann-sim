% network
clc;
clear;

%% Testing parameters
run_stats = false;      % true -> Run all nonlinearities described by CONT, false -> Run nonlinearity described by run_select
run_select = 1;         % selects a SINGLE nonlinear function (only active when run_stats is false)
N_RUNS    = 32;          % number of runs per nonlinearity, used for averaging (only active when run_stats is true)
N_LAYERS  = 1;          % number of layers used for network
CONT = [1 5];         % Indicies used for running nonlinearities (only active when run_stats is true)
allow_above_saturation = false;     % allows for above saturation amount (defined by M)
remove_dut             = true;     % removes dut (nonlinearity under test), useful for linearizing the network
use_nonlinearity       = false;     % removes ending nonlinearity (sigmoid, softmax, relu)
only_linear            = true;     % uses network with only linear response (only with N_LAYERS = 1 for now)

if run_stats == false
    CONT = 1;
end

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
SNCV = CV.fn(CV.W, 1);
PPS = {
    getpfit(X, @(X)((1/M).*X), 1),...
    getpfit(X, @(X)(((1/M).*X).^2), 6),...
    @(X)(CV.fn(CV.W, X)),...
    @(X)(CV.fn(CV.W-[CV.W(1) 0 0 0], X)),...
    @(X)(CV.fn(CV.W, X)-SCV)/SSCV,...
    @(X)(CV.fn(CV.W, X)/SNCV),...
};

PPSName = [
    "X",...
    "X^2",...
    "Fitted Nonlinear Curve",...
    "Shifted Fitted Nonlinear Curve",...
    "Normalized Scaled Fitted Nonlinear Curve",...
    "Normalized Fitted Nonlinear Curve",...
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
learnRate     = 1e-5;
numEpochs     = 16;
miniBatchSize = 64;

ss = [size(imread(cell2mat(testingData.Files(1)))), 1];
kernel = abs(randn(ss));
lvalue=1e-250;
c1 = 0.0;
c2 = 0.0;
S_AMT = M;

if allow_above_saturation
    S_AMT = 1000 * M;
end


%% create the network layers
ppContainer(length(PPS))=Container();
for j=CONT
    pp2=PPS{j};
    if ~isa(pp2, 'function_handle')
        dd2=polyder(pp2);
    end

    if PPSName(j) == "X"
        remove_dut = true;      % only allow remove when dut is not tested (i.e., linear)
    else
        remove_dut = false;
    end

    if remove_dut && only_linear
        only_linear = true;
    else
        only_linear = false;
    end

    ppContainer(j).Id = PPSName(j);

    inputLayer     = imageInputLayer(ss, Name='input', Normalization='rescale-zero-one');
    kernelLayer    = fullyConnectedLayer(32, Name='kernel');
    SShape         = flattenLayer(Name='sshape');
    PKernel        = CustomPositiveFullyConnectedLayer('kernel', 784, 32);
    protect1       = CustomNaNPreventionLayer('protect1', lvalue);
    positiveLayer  = CustomPositiveLayer('post1');
    add1           = CustomConstantAddLayer('add1', c1);
    sat1           = CustomSaturationLayer('sat1', M);

    % DUT           = reluLayer(Name='dut');
    A1            = CustomAnalyzerLayer('a1', S_AMT);

    if ~isa(pp2, 'function_handle')
        DUT = CustomPolynomialNonLinearLayer('dut',pp2,dd2,ss,1,1);
    else
        DUT = functionLayer(pp2, Name='dut');
    end

    % second linear layer
    L1            = fullyConnectedLayer(30, Name='L1');

    % force weights to be positive
    positiveLayer2 = CustomPositiveLayer('post2');
    add2          = CustomConstantAddLayer('add2', c2);
    sat2          = CustomSaturationLayer('sat2', M);

    A2            = CustomAnalyzerLayer('a2', S_AMT);

    if ~isa(pp2, 'function_handle')
        DUT2 = CustomPolynomialNonLinearLayer('dut2', pp2, dd2, ss, 1, 1);
    else
        DUT2 = functionLayer(pp2, Name='dut2');
    end
       
    flatten       = fullyConnectedLayer(10, Name='flatten');
    %L2            = softmaxLayer(Name='L2');
    %L2            = sigmoidLayer("Name","L2");
    L2            = reluLayer(Name="L2");           % only nonlinearity is DUT (notice that relu is a nonlinear function, 
                                                    % but since all values are
                                                    % guaranteed to be
                                                    % positive, then relu
                                                    % is simply linear
    %classifyy     = classificationLayer(Name='classify');
    classifyy = CustomMSEClassificationLayer('classify');
    
    if N_LAYERS == 2
        layers = [
           inputLayer       % simulates optical side
           kernelLayer
           protect1
           positiveLayer
           add1
           
           %A1
           L1
           positiveLayer2
           A2
           add2
    
    
           flatten          % digital side
           classifyy
        ];
    else
        layers = [
          inputLayer
          kernelLayer
          protect1
          positiveLayer
          add1

          flatten
          classifyy
        ];
    end

    if allow_above_saturation == false
        layers = [layers;sat1];
    end

    if remove_dut == false
        layers = [layers;DUT];
    end

    if N_LAYERS==2
        if remove_dut == false
            layers = [layers;DUT2];
        end
        if allow_above_saturation == false
            layers = [layers;sat2];
        end
    end

    if use_nonlinearity
        layers = [layers;L2];
    end

    if only_linear
        layers = [
            inputLayer
            kernel
            post1
            flatten
            classifyy
        ];
    end
    
    %% connect Layers
    lgraph = layerGraph();
    for i=1:length(layers)
        lgraph = addLayers(lgraph, layers(i));
    end

    if ~only_linear  
        lgraph = connectLayers(lgraph, 'input', 'kernel');
        lgraph = connectLayers(lgraph, 'kernel', 'protect1');
        lgraph = connectLayers(lgraph, 'protect1', 'post1');
    
        lgraph = connectLayers(lgraph, 'post1', 'add1');
    
        if remove_dut == false
            if allow_above_saturation
                lgraph = connectLayers(lgraph, 'add1', 'dut');
            else
                lgraph = connectLayers(lgraph, 'add1', 'sat1');
                lgraph = connectLayers(lgraph, 'sat1', 'dut');
            end
        end
    
        if N_LAYERS == 2
            if remove_dut == false
                lgraph = connectLayers(lgraph, 'dut', 'L1');
                lgraph = connectLayers(lgraph, 'a2', 'dut2');
                lgraph = connectLayers(lgraph, 'dut2', 'flatten');
            else
                if allow_above_saturation == false
                    lgraph = connectLayers(lgraph, 'add1', 'sat1');
                    lgraph = connectLayers(lgraph, 'sat1', 'L1');
                end
                lgraph = connectLayers(lgraph, 'a2', 'flatten');
            end
            if allow_above_saturation
                if remove_dut
                    lgraph = connectLayers(lgraph, 'add1', 'L1');
                end
                lgraph = connectLayers(lgraph, 'add2', 'a2');
            else
                lgraph = connectLayers(lgraph, 'add2', 'sat2');
                lgraph = connectLayers(lgraph, 'sat2', 'a2');
            end
            lgraph = connectLayers(lgraph, 'L1', 'post2');
            lgraph = connectLayers(lgraph, 'post2', 'add2');
        else
            if remove_dut == false
                lgraph = connectLayers(lgraph, 'dut', 'flatten');
            else
                if allow_above_saturation
                    lgraph = connectLayers(lgraph, 'add1', 'flatten');
                else
                    lgraph = connectLayers(lgraph, 'add1', 'sat1');
                    lgraph = connectLayers(lgraph, 'sat1', 'flatten');
                end
            end
        end
    
        if use_nonlinearity
            lgraph = connectLayers(lgraph, 'flatten', 'L2');
            lgraph = connectLayers(lgraph, 'L2', 'classify');
        else
            lgraph = connectLayers(lgraph, 'flatten', 'classify');
        end
    else
        lgraph = connectLayers(lgraph, 'input', 'kernel');
        lgraph = connectLayers(lgraph, 'kernel', 'post1');
        lgraph = connectLayers(lgraph, 'post1', 'flatten');
        lgraph = connectLayers(lgraph, 'flatten', 'classify');
    end
    
    if run_stats == false
        figure;
        plot(lgraph);
    end
    
    %% give options to the network
    
    if run_stats == false
    options = trainingOptions('sgdm',...
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
        options = trainingOptions('sgdm',...
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
        for i=CONT
            cc = ppContainer(i);
            plot(cc.acc.*100);
            yline(cc.avg*100);
            yl=cc.Id+" Average";
            lgnds=[lgnds;cc.Id;yl];
        end
    hold off;
    title("Accuracy vs Simulation Count");
    xlabel("Simulation Count");
    ylabel("Accuracy");
    legend(lgnds);
end

CI = ContainerInspector(ppContainer);