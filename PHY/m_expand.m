function [y, r] = m_expand(x, N)
    l = length(x);
    y = zeros(1,N);
    r = floor(N/l); % each pixel of x takes r pixels of N
    c = 1;
    for i=1:l
        pix = x(i);
        for k=1:r
            if k>8 && k<r-8
                y(c)=pix;
            end
            c = c + 1;
        end
        if c > N
            break;
        end
    end
end