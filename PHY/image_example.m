clc
clear
ip_addr='129.237.123.147';
port   = 5000;
src=2;
dst=1;
ffreq=7400;
rp_awg = RP_AWG(ip_addr, port, ffreq, 1, src, dst);
rp_awg.reset();
rate = 125e6;

N = 16383;  % number of samples in buf fer

% set trigger parameters
rp_awg.en_trig = true;      % we want to use the trigger
rp_awg.trig_lvl = 0.78;      % set the trigger level (reads when threshold is crossed)
rp_awg.trig_del = ceil(N/2)-104;      % set the delay (using N/2 of samples seems to be the best)

rp_awg.setup_adc();

vstart = 300;
vend   = 500;
[z, vstart, vendr, owidth] = create_signal_base(N, vstart, vend);
[ostart, ~] = preamble_props(z, rp_awg.trig_lvl);
l=vendr-vstart;

% hone in on frequency
%rp_awg = freq_hone(rp_awg, ostart, owidth, z, 1);
rp_awg.dac_freq


TT=9;
% load the data
dataTest = create_imagedatastore('Images/t10k-labels-idx1-ubyte.gz','TestImagesPNG/');
NNN = find(dataTest.Labels==categorical(TT));

im1 = double(imread(cell2mat(dataTest.Files(NNN(2)))));

% normalization
im1 = im1/255;

% multiply signal by random amount
im1 = im1.*abs(randn(size(im1)));
im1 = im1 ./ max(im1,[],'all');
im1(im1>0)=im1(im1>0)+0.2;

figure;
imagesc(im1);
colorbar();
title("Transmitted Image");

% 
% figure;
% imagesc(im1);
% colorbar();
% title("Input image");

% reshape to vector
vv = reshape(im1.',1,[]);
vv = vv*0.7;

% now, we remove zeros from beginning and end
vf = find(vv~=0);
vv = vv(vf(1):vf(end));

% send the signals in two parts
sep=floor(length(vv)/2);
vfirst  = vv(1:sep);
vsecond = vv(sep+1:end);
vu = [vfirst, vsecond];

% expand the signal
sep=500;
[mfirst, rfirst] = m_expand(vfirst, l-sep);
[msecond, rsecond] = m_expand(vsecond, l-sep);
mfirst = [zeros(1,sep) mfirst];
msecond= [zeros(1,sep) msecond];

z(vstart:vendr-1)=mfirst;
filterC=exp(-1*(linspace(-0.5,0.5,20)/0.6).^2);
q=filter(filterC,1,z);
q=filter(filterC,1,q);
%q=movmean(z,5);
%q=z;
q=q./max(q);
q=2*q-1;

figure;
plot(q);
in1=q;
title("Sent signal (first)");

rp_awg.transmit(q);
rp_awg.wait(3*N);
%y1=rp_awg.receive();
y1=rp_awg.adc_read();

figure;
plot(y1);
title("From ADC (first)");

% transmit second signal

z(vstart:vendr-1)=msecond;
q=filter(filterC,1,z);
%q=movmean(z,5);
q=q./max(q);
q=2*q-1;
in2=q;

% figure;
% plot(q);
% title("Sent signal (second)");

rp_awg.transmit(q);
rp_awg.wait(3*N);
%y2=rp_awg.receive();
y2=rp_awg.adc_read();

% figure;
% plot(y2);
% title("From ADC (second)");
% 
% smooth both signals
[pp, vv] = signal_retreive(rp_awg, y1, y2, 10, ostart, owidth, rfirst, rsecond, in1, in2, im1);

r1 = reshape(pp,[],28);

figure;
imagesc(r1);
colorbar();
title("Recovered Image");


