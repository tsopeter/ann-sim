% creat imags
clear;
filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

create_images(filenameImagesTrain, "TrainImagesPNG");
create_images(filenameImagesTest , "TestImagesPNG");

clear;
