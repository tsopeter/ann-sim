function ndigits = numdigits(x)
%  NDIGITS = NUMDIGITS(X)
%  A simple convenience tool to count the number of digits in
%  the integer part of a number.  For complex inputs, the result 
%  is the number of digits in the integer part of its magnitude.
%
%  X is a numeric scalar or array of any class or sign. 
%
%  Note that the integer 0 is considered to have zero digits.  
%  Consequently, for numbers on the interval -1<X<1, NDIGITS is 0.

% souce
% https://www.mathworks.com/matlabcentral/answers/10795-counting-the-number-of-digits#:~:text=function%20ndigits%20%3D%20numdigits%20%28x%29%20%25%20NDIGITS%20%3D,tool%20to%20count%20the%20number%20of%20digits%20in
ndigits = ceil(log10(abs(double(fix(x)))+1));
end