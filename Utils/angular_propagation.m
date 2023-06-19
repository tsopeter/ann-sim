function H = angular_propagation(Nx, Ny, nx, ny, wv, d)
    dx = nx/Nx;
    dy = ny/Ny;
    k = 2 * pi/wv;

    zx = linspace(-1/(2*dx),1/(2*dx),Nx);
    zy = linspace(-1/(2*dy),1/(2*dy),Ny);
    kx = 2*pi*zx;
    ky = 2*pi*zy;

    [Kx, Ky] = meshgrid(kx, ky);

    H = exp(1i * k * d * sqrt(1 - (wv*Kx).^2 - (wv*Ky).^2));
end