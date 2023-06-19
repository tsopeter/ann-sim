clc
clear
IP='129.237.123.147';
PORT=5000;
SRC=2;
DST=1;

sshift=-100;
locs_rem=0;
peak_height=0.8;

% D1/D2=1.0738

[pp, UU] = get_transfer(IP, PORT, DST, SRC, sshift, locs_rem, peak_height);
x  = linspace(0, 1, 256);
y  = polyval(pp, x);

figure;
plot(x,y);