function [prestart, prewidth] = preamble_props(sig, r)
        s = sig(1,1:floor(end/2));
        t = find(s>=r);
        prewidth=numel(t);
        prestart=t(1);
    end