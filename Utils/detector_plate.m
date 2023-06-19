function output = detector_plate(Nx, Ny, nx, ny, radius1, radius2, lvalue)
    % remember Nx is columns, Ny is rows
    % nx and ny describe similarly but in meters, rather than arbitrary
    % numbers

    % design the detector plate

    plate = zeros(Ny, Nx);

    % at every 36 degrees, put a circle of radius2 at radius1
    for r=0:36:324
        plate = plate + circle_at(Nx, Ny, nx, ny, radius1, 0, radius2, lvalue);
        plate = imrotate(plate, 36, 'crop');
    end
    output = plate;
end