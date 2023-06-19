function Z = detector_values(X, Nx, Ny, nx, ny, r1, r2, lvalue)
    Z = zeros(1, 10, 'single');
    for r=0:9
        plate = circle_at(Nx, Ny, nx, ny, r1, 0, r2, lvalue);
        plate = imrotate(plate, 36*r, 'crop');
        s = sum(sum(X.*plate));
        Z(r+1)=s;
    end
end