function [pp] = getpfit(X, f, N)
%getpfit - get polynomial fit of function
%   computes the function Y=f(X) and fits Y to N polynomials
    Y=f(X);
    pp = polyfit(X,Y,N);
end