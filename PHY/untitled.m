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
rp_awg.trig_lvl = 0.8;      % set the trigger level (reads when threshold is crossed)
rp_awg.trig_del = ceil(N/2)-104;      % set the delay (using 2*N of samples seems to be the best)
    

rp_awg.setup_adc();

vstart = 500;
vend   = 500;
for i=1:10
    [z, vstart, vendr, owidth] = create_signal_base(N, vstart, vend);
    [ostart, ~] = preamble_props(z, rp_awg.trig_lvl);
    l=vendr-vstart;
    
    % hone in on frequency
    %rp_awg = freq_hone(rp_awg, ostart, owidth, z, 1);
    
    % load the data
    dataTest = load('Images\dataTest.mat').dataTest;
    im1 = double(imread(cell2mat(dataTest.Files(1))));
    
    % normalization
    im1 = im1/255;
    
    % multiply signal by random amount
    im1 = im1.*abs(randn(size(im1)));
    im1 = im1 ./ max(im1,[],'all');
    
    % 
    % figure;
    % imagesc(im1);
    % colorbar();
    % title("Input image");
    
    % reshape to vector
    vv = reshape(im1.',1,[]);
    vv = vv*0.75;
    
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
    filterC=exp(-1*(linspace(-0.5,0.5,25)/0.5).^2);
    q=filter(filterC,1,z);
    %q=movmean(z,5);
    %q=z;
    q=q./max(q);
    
    % figure;
    % plot(q);
    in1=q;
    % title("Sent signal (first)");
    
    rp_awg.transmit(q);
    rp_awg.wait(N);
    %y1=rp_awg.receive();
    y1=rp_awg.adc_read();
    
    % figure;
    % plot(y1);
    % title("From ADC (first)");
    
    % transmit second signal
    
    z(vstart:vendr-1)=msecond;
    q=filter(filterC,1,z);
    %q=movmean(z,5);
    q=q./max(q);
    in2=q;
    
    % figure;
    % plot(q);
    % title("Sent signal (second)");
    
    rp_awg.transmit(q);
    rp_awg.wait(N);
    %y2=rp_awg.receive();
    y2=rp_awg.adc_read();
    
    % figure;
    % plot(y2);
    % title("From ADC (second)");
    % 
    % smooth both signals
    %signal_retreive(rp_awg, y1, y2, 10, ostart, owidth, rfirst, rsecond, in1, in2);
end