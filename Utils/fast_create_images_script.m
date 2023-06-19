% creat imags
clear;
F=[128 128];
N=[256 256];
filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

fast_create_images(filenameImagesTrain, "DLPTrainImagesPNG", F, N);
fast_create_images(filenameImagesTest , "DLPTestImagesPNG",  F, N);

clear;