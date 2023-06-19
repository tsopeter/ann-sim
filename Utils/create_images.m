%
% create images

function create_images(filename, location)
    X = processImagesMNIST(filename);
    extn   = ".png";
    len = size(X, 4);
    nZeros = numdigits(len);
    for i=1:len
        Y=X(:,:,1,i);
        q = pad(string(i), nZeros, 'left', '0');
        q = q+extn;
        name = location+"/"+q;
        imwrite(Y, name);
    end
end