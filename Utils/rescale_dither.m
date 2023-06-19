function z = rescale_dither(img, F, M)
    %   img <- image to process
    %   F   <- size of rescaled 'pixel'
    %   M   <- highest level of pixel (typ. 255)
    %
    %   Methodology
    %       Each pixel i, j in image
    %       gets rescaled to a corresponding F
    %
    
    nF = F(1) * F(2);
    
    %   We need to map nF -> max img
    %   via a scaling. This scaling
    %   "sc" is sc = max(img)/nF
    %

    sc = nF/M;

    Qs = [size(img, 1) * F(1), size(img, 2) * F(2)];
    z = zeros(Qs, 'single');

    for x=1:size(img, 1)
        for y=1:size(img, 2)
            mask = spiral(F(1));
            t=single(img(x, y));
            v=round(sc*t);
            mask(mask<=v)=1;
            mask(mask>v)=0;
            xt=F(1)*x-(F(1)-1);
            yt=F(2)*y-(F(2)-1);
            z(xt:xt+F(1)-1,yt:yt+F(2)-1)=mask;
        end
    end
end