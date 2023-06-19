classdef CustomPhaseKernelLayer < nnet.layer.Layer %  & nnet.layer.Acceleratable 
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
        phi
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
        function layer = CustomPhaseKernelLayer(Name, kernel, rate)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = 1;
            layer.NumOutputs = 2;
            layer.kernel = kernel;
            layer.phi = dlarray(angle(kernel));
            layer.rate = rate;
        end
        
        function [Z1, Z2] = predict(layer,X)
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
            Z1 = X .* cos(layer.phi);
            Z2 = X .* sin(layer.phi);
        end

        function display_kernel_magnitude(layer)
            figure;
            imagesc(extractdata(abs(cos(layer.phi)+1i*sin(layer.phi))));
        end

        function display_kernel_angle(layer)
            figure;
            imagesc(extractdata(layer.phi));
        end

%         function [dLdX, dLdW1, dLdW2] = backward(layer, X, Z1, Z2, dLdZ1, dLdZ2, dLdSout)
%             % (Optional) Backward propagate the derivative of the loss
%             % function through the layer.
%             %
%             % Inputs:
%             %         layer   - Layer to backward propagate through 
%             %         X       - Layer input data 
%             %         Z       - Layer output data 
%             %         dLdZ    - Derivative of loss with respect to layer 
%             %                   output
%             %         dLdSout - (Optional) Derivative of loss with respect 
%             %                   to state output
%             %         memory  - Memory value from forward function
%             % Outputs:
%             %         dLdX   - Derivative of loss with respect to layer input
%             %         dLdW   - (Optional) Derivative of loss with respect to
%             %                  learnable parameter 
%             %         dLdSin - (Optional) Derivative of loss with respect to 
%             %                  state input
%             %
%             %  - For layers with state parameters, the backward syntax must
%             %    include both dLdSout and dLdSin, or neither.
%             %  - For layers with multiple inputs, replace X and dLdX with
%             %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
%             %    the number of inputs.
%             %  - For layers with multiple outputs, replace Z and dlZ with
%             %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
%             %    number of outputs.
%             %  - For layers with multiple learnable parameters, replace 
%             %    dLdW with dLdW1,...,dLdWP, where P is the number of 
%             %    learnable parameters.
%             %  - For layers with multiple state parameters, replace dLdSin
%             %    and dLdSout with dLdSin1,...,dLdSinK and 
%             %    dLdSout1,...,dldSoutK, respectively, where K is the number
%             %    of state parameters.
% 
%             % Define layer backward function here.
%             
%             W = size(X);
%             if length(W)<=2
%                 W(3)=1;
%                 W(4)=1;
%             end
%             
%             dLdW1 = zeros(size(layer.real_kernel), 'like', dLdZ1);
%             dLdW2 = zeros(size(layer.imag_kernel), 'like', dLdZ2);
% 
%             for i=1:W(4)
%                 dLdW1 = dLdW1 + (dLdZ1(:, :, 1, i) .* X(:,:,1,i) * layer.rate/W(4));
%                 dLdW2 = dLdW2 + (dLdZ2(:, :, 1, i) .* X(:,:,1,i) * layer.rate/W(4));
%             end
%             dLdX = dLdZ1;
%         end
    end
end