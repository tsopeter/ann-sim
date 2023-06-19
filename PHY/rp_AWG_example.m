%
ip_addr='129.237.123.147';
port   = 5000;
src=2;
dst=1;
rp_awg = RP_AWG(ip_addr, port, 10000, 1, src, dst);
rp_awg.reset();

N = 16383;  % number of samples in buffer

% set trigger parameters
rp_awg.en_trig = true;      % we want to use the trigger
rp_awg.trig_lvl = 0.4;      % set the trigger level (reads when threshold is crossed)
rp_awg.trig_del = 2*N;      % set the delay (using 2*N of samples seems to be the best)

N = 16383;  % number of samples in buffer
z = zeros(1, N);
z = m_sawtooth(N,4096,0,0.5);

% place a 1000 sample LOW
% z(1:1000)=0;
% 
% % place a 100 sample HIGH
% z(1001:1001+100-1)=1;
% 
% % then place our signal
% l=length(z(2001:end-2000));
% 
% f=1;
% t=linspace(0, 2*pi, l);
% x=sin(t)*0.6;
% z(2001:end-2000)=x;
% 
% % end with a 2000 sample LOW
% z(end-2000+1:end)=0;

figure;
plot(20*log10(abs(z)));

rp_awg.transmit(z);
rp_awg.wait(N);
y=rp_awg.receive();

figure;
plot(20*log10(abs(y)));
title("from arbitrary function @ f=10kHz");

%trigger;