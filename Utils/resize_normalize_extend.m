function z = resize_normalize_extend(X, Nx, Ny, k, lvalue, mx)
    W = size(X);

    if length(W)<=2
        W(3)=1;
        W(4)=1;
    end

    z = zeros(Nx, Ny, W(3), W(4), 'like', X);

    for i=1:W(4)
        Q = mask_resize(interp2(X(:,:,1,i), k)/mx, Nx, Ny);
        Q(Q==0)=lvalue;
        z(:,:,1,i) = Q;
    end
end