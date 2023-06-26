%% MMSE fitting
clc;
clear;

% load the data
ydata = load('NETWORK/tf.mat').uu3;
ydata = movmean(ydata, 10);
xdata = linspace(0, 1, length(ydata)).';
fn=@(x,xdata)(x(4).*(xdata-x(1)).*exp(-x(2)./(1+((xdata-x(1))./x(3)))));

w=[-0.3 1.2 0.7 1];
W=lsqcurvefit(fn,w,xdata,ydata);

figure;
plot(xdata, ydata, xdata, fn(W,xdata), xdata, fn(w, xdata));
legend("Input data", "Fitted Curve", "First fit");

figure;
plot(xdata(1:end-1), diff(fn(W,xdata))*(length(xdata)), xdata(1:end-1), diff(fn(w, xdata))*(length(xdata)));
legend("Fitted Curve", "First fit");