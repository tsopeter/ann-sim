function [pp, UU] = get_transfer(IP, PORT, DST, SRC, sshift, locs_rem, peak_height)
    N    = 16383;
    freq = 8192;
    awg  = RP_AWG(IP, PORT, freq, 1, SRC, DST);

    N_PEAKS=8;

    saw = 0:1:(N-1);
    n   = (N+1)/N_PEAKS;
    saw = mod(saw, n);
    saw = saw ./ max(saw);

    figure;
    plot(saw);
    title("input sawtooth wave");

    awg.en_trig = true;
    awg.trig_lvl = 0.2;
    awg.trig_del = ceil(N/2)-1;

    awg.setup_adc();
    awg.transmit(2*saw-1);
    awg.wait(N);
    y1=awg.adc_read();

    figure;
    plot(y1);
    title("retrieved sawtooth wave");

    y1(y1<0)=0;     % we want to get rid of the negative values caused by "ringing" (sharp freqeuency changes)

    y2=movmean(y1,100);
    y2=circshift(y2,sshift);
    y2(y2<peak_height)=0;
    [~,locs]=findpeaks(y2);

    y1=circshift(y1,sshift);

    N22=locs(end-locs_rem);
    N21=locs(1);

    N11=n;
    N12=N;
    

    y1=y1(N21:N22); % chop off the first instance
    y1=circshift(y1,sshift);
    x1=saw(N11:N12);

    D=(N12-N11)/(N22-N21);

    t1=(0:length(x1)-1)*8e-9;
    t2=(0:length(y1)-1)*D*8e-9;

    x2=interp1(t1,x1,t2);

    figure;
    plot(t2,[x2.',y1]);
    title("interpolated x1 signal compared to y1");


    y3=movmean(y1,100);
    y3(y3<peak_height)=0;
    [~,locs_y]=findpeaks(y3);
    [~,locs_x]=findpeaks(x2);

    diff=locs_x(1)-locs_y(1);
    y1=circshift(y1,floor(diff/2));

    figure;
    plot([x2.',y1]);

    diff=rem(length(y1),N_PEAKS-1);
    uu=reshape(y1(floor(diff/2):end-ceil(diff/2)-1),[],N_PEAKS-1);

    UU=mean(uu.').';
    UU=circshift(UU,-150);
    UU=UU(1:end-150);

    Z=linspace(0,1,length(UU));
    pp=polyfit(Z,UU,6);

    figure;
    plot(UU);

end