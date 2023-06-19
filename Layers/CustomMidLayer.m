classdef CustomMidLayer < nnet.layer.Layer %  & nnet.layer.Acceleratable 
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        rate;
        kernel;
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
        W1;
        W2;
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = CustomMidLayer(Name, kernel, rate)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = 2;
            layer.NumOutputs = 2;
            layer.kernel = kernel;
            layer.W1 = dlarray(real(kernel));
            layer.W2 = dlarray(imag(kernel));
            layer.rate = rate;
        end
        
        function [Z1, Z2] = predict(layer,X, Y)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            % normalize and apply layer weights
            Z1 = X .* layer.W1 - Y .* layer.W2;
            Z2 = X .* layer.W2 + Y .* layer.W1;
        end

        function display_kernel(layer)
            figure;
            imagesc(extractdata(abs(layer.W1+1i*layer.W2)));
        end
    end
end