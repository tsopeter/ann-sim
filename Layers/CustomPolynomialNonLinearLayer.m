classdef CustomPolynomialNonLinearLayer < nnet.layer.Layer
    properties
        %
        % polynomial fitted data
        P0

        %
        % derivative of polynomial fitted data
        D0

        %
        % Needed properties
        ss
        xnormal
        ynormal
    end

    %
    % Unused properties, the Custom Polynomial Layer will not
    % learn
    properties (Learnable)
    end
    properties (State)
    end
    properties (Learnable, State)
    end

    methods
        function layer = CustomPolynomialNonLinearLayer (Name, poly, der, ss, xnormal, ynormal)
            layer.Name       = Name;
            layer.NumInputs  = 1;
            layer.NumOutputs = 1;
            layer.ss         = ss;

            layer.P0         = poly;
            layer.D0         = der;
            
            layer.xnormal    = xnormal;
            layer.ynormal    = ynormal;
        end

        function Z = predict(layer, X)
            Z = polyval(layer.P0, X*layer.xnormal)/layer.ynormal;
        end

        function dLdX = backward(layer, X, Z, dLdZ, dLdSout)
            dZdX = polyval(layer.D0*layer.xnormal, X*layer.xnormal)/layer.ynormal;
            dLdX = dLdZ .* dZdX;
        end
    end
end