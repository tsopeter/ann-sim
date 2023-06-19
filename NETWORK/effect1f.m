function Y = effect1f(X)
    Y = X ./ max(max(X, [], [28 1]));
    Y(Y>0)=Y(Y>0)+0.2;
    Y = Y * 0.7;
end
