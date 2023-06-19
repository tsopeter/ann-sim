x = linspace(-3, 3, 1024);
u = x*0;
u(x>-2)=1;
u(x>2)=0;
figure;
plot(u);

z=exp(-x.^2);

figure;
plot(z);

m=xcorr(u,z);

figure;
plot(m);