%   combines and recovers the signal
%   head <- first half of transferred signal
%   tail <- second half of transferred signal
%   ostart <- start index of preamble of original signal
%   owidth <- width of preamble of original signal
%   r1 <- pixel ratio of first signal (head)
%   r2 <- pixel ratio of second signal (tail)
function [out, vv] = signal_retreive(awg, head, tail, N, ostart, owidth, r1, r2, in1, in2, img)
    

    % remove high frequency
    fhead = fftshift(fft(head));
    ftail = fftshift(fft(tail));

    fhead(1:5000)=10e-20;
    fhead(13000:end)=10e-20;
    ftail(1:5000)=10e-20;
    ftail(13000:end)=10e-20;

    head = real(ifft(ifftshift(fhead)));
    tail = real(ifft(ifftshift(ftail)));

    % remove noise floor
    head(head<=0.096)=0;
    tail(tail<=0.096)=0;    

    figure;
    plot(head);

    figure;
    plot(tail);

    % create a moving average filter and
    % first, smooth out the response and ripples caused by ADC
    filterC=ones(1,N);
    headn = filter(filterC, 1, head);
    tailn = filter(filterC, 1, tail);
    

    % normalize
    headn = headn./max(headn);

    % get rid of preamble and ending
    headn = headn(300:end-50);
    headn(headn<0.1)=0;
    [~,hlocs]=findpeaks(headn,"MinPeakWidth",17);
    z1=zeros(length(headn),1);
    z1(hlocs)=headn(hlocs);

    tailn = tailn./max(tailn);
    tailn = tailn(300:end-50);
    tailn(tailn<0.1)=0;
    [~,tlocs]=findpeaks(tailn,"MinPeakWidth",17);
    z2=zeros(length(tailn),1);
    z2(tlocs)=tailn(tlocs);

    figure;
    plot([headn,z1]);
    title("smoothed head");

    figure;
    plot([tailn,z2]);
    title("smoothed tail");

    combined=[headn;tailn];
    [~,clocs]=findpeaks(combined, "MinPeakWidth",17);

    vv = combined;
    out = zeros(numel(img),1);
    out(img>0)=combined(clocs);
end