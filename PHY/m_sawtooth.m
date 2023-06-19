function z = m_sawtooth(N, m, bias, k)
    z = 0:1:(N-1);
    z = (2*k*mod(z, m)/m)-bias;
end