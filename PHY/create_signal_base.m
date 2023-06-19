function [z, vstart, vendr, owidth] = create_signal_base(N, vstart, vend)
    z = zeros(1, N);
    % for the preamble, set the first 100 to vstart-100 as HIGH
    z(100:vstart-100+1)=1;
    owidth=length(z(100:vstart-100+1));
    vendr = length(z)-vend;
end