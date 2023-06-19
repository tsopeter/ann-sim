function Y = effect2f(X)
    Y = X / 0.7;
    Y(Y>0)=Y(Y>0)-0.2;
end