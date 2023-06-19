classdef CustomFastPropagationLayer < nnet.layer.Layer %  & nnet.layer.Acceleratable 
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        w
        wc
        Nx
        Ny
        nx
        ny
        d
        wv
        scale
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
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
        function layer = CustomFastPropagationLayer(Name, Nx, Ny, nx, ny, d, wv)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = 1;
            layer.NumOutputs = 1;

            layer.Nx = Nx;
            layer.Ny = Ny;
            layer.nx = nx;
            layer.ny = ny;
            layer.d  = d;
            layer.wv = wv;
            layer.scale = 1/(sqrt(2*pi));
            layer = layer.compute_w();
        end

        function layer = compute_w(layer)
            k=2*pi/layer.wv;
            f_dx = layer.Nx/layer.nx;
            f_dy = layer.Ny/layer.ny;

            posx = linspace(-f_dx/2, f_dx/2-layer.Nx, layer.Nx);
            posy = linspace(-f_dy/2, f_dy/2-layer.Ny, layer.Ny);
            [fxx, fyy] = meshgrid(posx, posy);
            a = layer.wv .* fxx;
            b = layer.wv .* fyy;
            g = a.^2+b.^2;
            t = sqrt(1-a.^2-b.^2);
            g(g>1)=NaN;
            g(g<=1)=t(g<=1);
            g(isnan(g))=0;
            layer.w = exp(1i*k*g*layer.d)*layer.scale;
            layer.wc = layer.w';
        end
        
        function Z = predict(layer, X)
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
            Z = zeros(size(X), 'like', X);
            for i=1:size(X, 4)
                fp1 = fft(real(X(:,:,1,i))).';
                fp2 = fft(fp1).';
                fp3 = ifftshift(fftshift(fp2) .* layer.w);
                zp1 = ifft(fp3).';
                zp2 = ifft(zp1).';
                % Z(:,:,1,1)=abs(zp2);
                Z(:,:,1,i)=real(zp2.*conj(zp2));
            end
        end

        function dLdX = backward(layer, X, Z, dLdZ, dLdSout)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer   - Layer to backward propagate through 
            %         X       - Layer input data 
            %         Z       - Layer output data 
            %         dLdZ    - Derivative of loss with respect to layer 
            %                   output
            %         dLdSout - (Optional) Derivative of loss with respect 
            %                   to state output
            %         memory  - Memory value from forward function
            % Outputs:
            %         dLdX   - Derivative of loss with respect to layer input
            %         dLdW   - (Optional) Derivative of loss with respect to
            %                  learnable parameter 
            %         dLdSin - (Optional) Derivative of loss with respect to 
            %                  state input
            %
            %  - For layers with state parameters, the backward syntax must
            %    include both dLdSout and dLdSin, or neither.
            %  - For layers with multiple inputs, replace X and dLdX with
            %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
            %    the number of inputs.
            %  - For layers with multiple outputs, replace Z and dlZ with
            %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
            %    number of outputs.
            %  - For layers with multiple learnable parameters, replace 
            %    dLdW with dLdW1,...,dLdWP, where P is the number of 
            %    learnable parameters.
            %  - For layers with multiple state parameters, replace dLdSin
            %    and dLdSout with dLdSin1,...,dLdSinK and 
            %    dLdSout1,...,dldSoutK, respectively, where K is the number
            %    of state parameters.

            % Define layer backward function here.
            dLdX = zeros(size(X), 'like', X);
            for i=1:size(X, 4)
                fp1 = fft(real(X(:,:,1,i))).';
                fp2 = fft(fp1).';
                fp3 = ifftshift(fftshift(fp2) .* layer.w);
                zp1 = ifft(fp3).';
                zp2 = ifft(zp1).';
                
                dLdh = 2 * dLdZ(:,:,1,i) .* zp2;

                dlfp1 = fft(dLdh).';
                dlfp2 = fft(dlfp1).';
                dlfp3 = ifftshift(fftshift(dlfp2) .* layer.wc);
                dlzp1 = ifft(dlfp3).';
                dlzp2 = ifft(dlzp1).';
                dLdX(:,:,1,i)=real(dlzp2);
            end
        end
    end
end