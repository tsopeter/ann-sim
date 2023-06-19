function Z = create_imagedatastore(labelfilename, location)
    Y = processLabelsMNIST(labelfilename);

    imds = imageDatastore(location);
    imds.Labels = Y;
    Z = imds;
end