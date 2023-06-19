function z = get_propagation_distance(Nx, Ny, nx, ny, distance, wavelength)
    dx = nx/Nx;     % size of each element
    dy = ny/Ny;
    
    rangex = 1/dx;  % number of frequencies available
    rangey = 1/dy;

    posx = linspace(-rangex/2, rangex/2, Nx);
    posy = linspace(-rangey/2, rangey/2, Ny);

    [fxx, fyy] = meshgrid(posy, posx);
    
    kz = 2 * pi * sqrt((1/wavelength)^2 -(fxx.^2)-(fyy.^2));

    z = exp(1i * kz * distance);
    z = ifft2(ifftshift(z));
end