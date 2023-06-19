function Y = query(awg, X, D, SHIFT)
    awg.transmit(X);
    awg.wait(length(X));
    y = awg.receive();

    y = circshift(y, SHIFT);
    
end