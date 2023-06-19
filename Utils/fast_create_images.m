function fast_create_images(filename, location, F, N)
    X = processImagesMNIST(filename);
    extn = '.png';
    len  = size(X, 4);
    nZeros = numdigits(len);
    for i=1:len
        Y = imresize(X(:,:,1,i), F, "nearest", "Antialiasing",true);
        Y = mask_resize(Y, N(1), N(2));
        q = pad(string(i), nZeros, 'left', '0');
        q = q+extn;
        name = location+"/"+q;
        imwrite(Y, name);
    end
end