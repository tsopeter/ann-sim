% compute the transfer function
clc;
clear;

X  = load('SAWTOOTH/input.mat').saw;

N  = length(X);

Y  = load('SAWTOOTH/received.mat').y1;

Y2 = circshift(Y, -25);

figure;
plot(0:1:N-1, X, 0:1:length(Y2)-1 ,Y2);

mavgY2 = movmean(Y2, 100);
[~, locsY2] = findpeaks(mavgY2);
[~, locsX]  = findpeaks(X);

N22=locsY2(end);
N21=locsY2(1);

N12=locsX(end);
N11=locsX(1);

X2 = X(N11:N12);
Y3 = Y2(N21:N22);

Y3 = circshift(Y3,-75);

figure;
plot(0:1:length(X2)-1, X2, 0:1:length(Y3)-1 ,Y3);

D = (N12-N11)/(N22-N21);

t1=(0:length(X2)-1)*8e-9;
t2=(0:length(Y3)-1)*D*8e-9;

X3=interp1(t1,X2,t2);

figure;
plot(t2,[X3.',Y3]);

[~,locsY3]=findpeaks(movmean(Y3, 100));
[~,locsX3]=findpeaks(X3);

diff=locsX3(1)-locsY3(1);
Y4=circshift(Y3,floor(diff/100));
Y4=circshift(Y4,27);

figure;
plot([X3.',Y4]);

diff=rem(length(Y4),6);
Y5=Y4(floor(diff/2):end-ceil(diff/2)-1);
uu=reshape(Y5,[],6);

uu2=uu(15:end,:);
uu3=mean(uu2.').';

figure;
plot(uu3);

Z=linspace(0,1,length(uu3));
pp=polyfit(Z,uu3,6);
