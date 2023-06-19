clc;
clear;
y1 = load('SAWTOOTH/transfer_function_raw.mat').UU;

% bias y1 to 0
y1 = y1 - min(y1);

x  = linspace(0, 1, length(y1));

pp = polyfit(x, y1, 4);

y2 = polyval(pp, x);

figure;
plot(x, [y1, y2.']);

y3 = polyval(pp, x+0.2);

figure;
plot(x+0.2, y3);