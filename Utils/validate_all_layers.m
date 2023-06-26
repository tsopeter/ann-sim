% layer validation
clc
clear

Name='name';
ss=[32, 32];
lval = 1e-10;

%% Add custom layers to Layers for validation
Layers = {
   {'CustomAmplitudeKernelLayer', Name, randn(ss), 1},...
   {'CustomAnalyzerLayer', Name, 3, 1},...
   {'CustomConstantAddLayer', Name, 0.2, 1},...
   {'CustomHProdLayer', Name, 2},...
   {'CustomNaNPreventionLayer', Name, lval, 1},...
   {'CustomPhaseKernelLayer', Name, exp(1i*randn(ss)), 1, 1},...
   {'CustomPolarizationLayer', Name, 2},...
   {'CustomPositiveLayer', Name, 1},...
};

%% Run validation code
results={};
for layer=Layers
    fn=(layer{1}{1});
    name=(layer{1}{2});
    % get the number of parameters (remove first two and last)
    numParams = length(layer{1})-3;

    S = repmat({ss}, layer{1}{end});
    S = S(1,:);

    if numParams <= 0
        % no parameters, just evaluate the function
        code=['checkLayer(', fn, '(', name, '), S)'];
        r = eval(code);
    else
        clear vars;
        % create variables as simple containers
        vars(numParams)=SimpleContainer();
        
        V = '';
        for i=1:numParams
            vars(i).x = layer{1}{2+i};
            V = [V, 'vars(',num2str(i),').x, '];
        end
        V=V(1:end-2);

        code = ['checkLayer(', fn, '(', name, ', ', V, '), S)'];
        r = eval(code);

    end
    % append results
    results = [results;{fn, r}];
    disp(numParams);
end

%% Print results
for i=1:length(Layers)
    name=(results{i,1});
    res =results{i,2};
    nFailed = res.Failed;
    nPassed = res.Passed;
    nIncomp = res.Incomplete;
    nTime   = res.Duration;
    R=[name, ', Passed: ', num2str(nPassed), ', Failed: ', num2str(nFailed), ', Incomplete: ', num2str(nIncomp), ', Time: ', num2str(nTime)];
    disp(R);
end