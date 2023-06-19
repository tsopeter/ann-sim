%
% sam_serial network
clc;
clear;

% network images
filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

dataTrain = create_imagedatastore(filenameLabelsTrain, "DLPTrainImagesPNG/");
dataTest  = create_imagedatastore(filenameLabelsTest , "DLPTestImagesPNG/");

dataTest  = partition(dataTest, 10, 1);

% network training parameters
learnRate     = 1e-4;
numEpochs     = 64;
miniBatchSize = 128;


% network topography parameters
ss     = [256 256 1];
kernel = abs(randn(ss));

nxx    = 1;
rxx1   = 1/3;
rxx2   = 1/15;
lval   = 1e-6;

%
% get the parameters
[P0, D0, t0] = SAM_ser_dataloader(4);
xnormal = 1e3;  % normalize the curve to approximately 1 (should be fine for now)
ynormal = 1;

%
%   [Input]->[Kernel]->[Positive]->[NonLinear]->[Flatten]->[SoftMax]->[Class]
%   
%

inputLayer     = imageInputLayer(ss, "Name", "input", "Normalization", "rescale-zero-one");
kernelLayer    = CustomAmplitudeKernelLayer("kernel", kernel);
positiveLayer  = CustomPositiveLayer('positive');

% normal ReLU layer, commonly used
DUT            = reluLayer('Name', 'dut');

% Layer under test
%DUT            = CustomPolynomialNonLinearLayer('dut', P0, D0, ss, xnormal, ynormal);


flatten        = CustomFlattenLayer(1, 'flatten', ss(1), ss(2), nxx, nxx, rxx1, rxx2, lval);
softmax        = softmaxLayer("Name", 'softmax');
classification = classificationLayer("Name", 'classification');

layers = [
        inputLayer
        kernelLayer
        positiveLayer

        DUT

        flatten
        softmax
        classification
];

lgraph = layerGraph();
for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input', 'kernel');
lgraph = connectLayers(lgraph, 'kernel', 'positive');

lgraph = connectLayers(lgraph, 'positive', 'dut');
lgraph = connectLayers(lgraph, 'dut', 'flatten');

lgraph = connectLayers(lgraph, 'flatten', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classification');

%figure;
%F=detector_plate(ss(1), ss(2), nxx, nxx, rxx1, rxx2, lval);
%imagesc(F);

options = trainingOptions('adam',...
    InitialLearnRate=learnRate,...
    MaxEpochs=numEpochs,...
    Shuffle='every-epoch',...
    ValidationData=dataTest,...
    ValidationFrequency=512,...
    Verbose=true,...
    Plots='training-progress',...
    ExecutionEnvironment='auto',...
    DispatchInBackground=false,...
    MiniBatchSize=miniBatchSize);

net = trainNetwork(dataTrain, lgraph, options);
YPred = classify(net,dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

