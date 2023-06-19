clc
clear

IMG=zeros([28, 10*28], 'double');

for i=0:9
    img=load(['NUMBERS/' num2str(i) '_tx.mat']).im1;
    IMG(:,28*i+1:28*(i+1))=img;
end

figure;
imagesc(IMG);
colorbar();
title("Input numbers");

for i=0:9
    img=load(['NUMBERS/' num2str(i) '_rx.mat']).r1;
    IMG(:,28*i+1:28*(i+1))=img;
end

figure;
imagesc(IMG);
colorbar();
title("Output numbers");