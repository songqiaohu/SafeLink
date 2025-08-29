classdef RVFL<handle
    properties
        % 参数
        Ne = 10;
        N2 = 10;
        enhence_function = 'leakyrelu';
        reg1 = 0.01;
        reg2 = 1.0;
        c_1 = 5.0;
        c_2 = 1.0;
        
        % 变量
        C;
        W;
        P;
        Q;
        K;
        Wh;
        bh;
        pesuedoinverse;
        normalscaler;
        enhence_generator;
        dimension;
        mean;
        std;

        L1;
        L2;
    end
    
    methods
        % 构造函数
        function obj = RVFL(Ne, N2, enhence_function, reg1, reg2, c_1, c_2)
            if nargin > 0
                obj.Ne = Ne;
                obj.N2 = N2;
                obj.enhence_function = enhence_function;
                obj.reg1 = reg1;
                obj.reg2 = reg2;
                obj.c_1 = c_1;
                obj.c_2 = c_2;
            end
            
            obj.W = 0;
            obj.P = 0;
            obj.K = 0;
            obj.C = [0, obj.c_1;obj.c_2, 0];
            obj.normalscaler = Scaler();
            obj.enhence_generator = node_generator();

            obj.L1 = 2;
            obj.L2 = 2;
        end
        
        % fit方法
        function obj = fit(obj, oridata, orilabel)
            obj.dimension = size(oridata, 2);
            data = obj.normalscaler.fit_transform(oridata);
            obj.mean = obj.normalscaler.Mean;
            obj.std = obj.normalscaler.Std;

            label = zeros(size(orilabel, 1), 2);
            for i=1:size(orilabel, 1)
                if orilabel(i)==-1
                    label(i,1) = 1;
                else if orilabel(i)==1
                    label(i, 2) = 1;
                end
                end
            end
            
            enhencedata = obj.enhence_generator.generator_nodes(data, obj.Ne, obj.N2, obj.enhence_function);
            obj.Wh = obj.enhence_generator.Wlist;
            obj.bh = obj.enhence_generator.blist;
            inputdata = [data, enhencedata];  % 拼接数据

            
            
            [r, ~] = size(inputdata' * inputdata);
            obj.pesuedoinverse = pinv(obj.reg2 * inputdata' * inputdata + obj.reg1 * eye(r));
            obj.W = obj.pesuedoinverse * (obj.reg2 * inputdata' * label - inputdata' * label * obj.C);
            obj.K = pinv(obj.reg2 * inputdata' * inputdata + obj.reg1 * eye(r));
            obj.Q = obj.reg2 * inputdata' * label - inputdata' * label * obj.C;
        end
        
        % softmax_normalization
        function softmax_array = softmax_norm(~, array)
            exp_array = exp(array);
            exp_array(isinf(exp_array)) = 1;
            sum_exp_array = sum(exp_array, 2);
            softmax_array = exp_array ./ (sum_exp_array + 1e-6);
        end
        
        % predict方法
        function result = predict(obj, testdata)
            logit = obj.predict_proba(testdata);
            result = 2 * logit * [0;1] - 1;
            for i=1:length(result)
                if result(i)>0
                    result(i)=1;
                else if result(i)<0
                        result(i)=-1;
                end
                end
            end
        end
        
        % predict_proba方法
        function result = predict_proba(obj, testdata)
            if size(testdata, 2)==1
                if length(testdata)==4
                testdata = testdata([1, 2])';
                else if length(testdata)==2
                        testdata = testdata';
                end
                end
            else 
                testdata = testdata(:,1:2);
            end

            testdata = obj.normalscaler.transform(testdata);
            test_inputdata = obj.transform(testdata);
            org_prediction = test_inputdata * obj.W;
            result = 2 * org_prediction * [0;1] - 1;
        end
        
        % 梯度计算
        function result = grad(obj, x)
            switch obj.enhence_function
                case 'sigmoid5'
                    result = 5*exp(-5*x) / (1 + exp(-5*x)).^2;
                case 'sigmoid3'
                    result = 3*exp(-3*x) / (1 + exp(-3*x)).^2;
                case 'sigmoid'
                    result = exp(-x) / (1+exp(-x)).^2;
                case 'tanh'
                    result = 1 - tanh(x)^2;
                case 'relu'
                    result = double(x > 0);
                case 'leakyrelu'
                    % result = double(x > 0) + 0.01 * double(x <= 0);
                    result = double(x >= 1) + 0.01 * double(x <= 0) + (15*(1-0.01)*x.^4-32*(1-0.01)*x.^3+18*(1-0.01)*x.^2+0.01).*double(x>0&x<1);
                case 'softplus'
                    result = 1 / (1 + exp(-x));
            end
        end
        
        % 二阶梯度计算
        function result = grad2(obj, x)
            switch obj.enhence_function
                case 'sigmoid5'
                    result = 25 * (-exp(-5*x) + exp(-10 * x)) / (1 + exp(-5*x))^3;
                case 'sigmoid3'
                    result = 9 * (-exp(-3*x) + exp(-6 * x)) / (1 + exp(-3*x))^3;
                case 'sigmoid'
                    result = (exp(-2*x)-exp(-x))/(1+exp(-x))^3;
                case 'relu'
                    result = 0;
                case 'leakyrelu'
                    % result = 0;
                    result = 0 * double(x > 1|x<=0) + (60*(1-0.01)*x.^3-96*(1-0.01)*x.^2+36*(1-0.01)*x).*double(x>0&x<1);
                case 'tanh'
                    result = -2 * tanh(x) * (1 - tanh(x)^2);
                case 'softplus'
                    result = exp(-x) / (1 + exp(-x))^2;
            end
        end
        













        %% Gradient_B方法
        function grad = gradient_B(obj, x)
            % x
            % if length(x)==4
            %     x = x([1, 2])';
            % else
            %     x = x(:, 1:2);
            % end
            x = x';
            W0 = obj.W(1:obj.dimension,:);


            W1 = obj.W(obj.dimension + 1:end,:);


            Wh_matrix = cell2mat(obj.Wh);

            gradient = 2 * W0* [0;1] ./ (obj.std + 1e-6)' ;





            A = 0;
            for i = 1:size(Wh_matrix, 2)
                Wh1 = Wh_matrix(:, i);

                bh1 = obj.bh{floor((i - 1) / obj.N2) + 1};
                A = A +  2*W1(i,:) * [0;1] * obj.grad(((x - obj.mean) ./ (obj.std + 1e-6)) * Wh1 + bh1)*Wh1 ./ (obj.std + 1e-6)';
            end
            grad = gradient + A;

            % h = 1e-5;
            % grad = zeros(size(x));  % 初始化梯度向量，大小与 x 一致
            % for i = 1:length(x)
            %     x_plus = x;  
            %     x_plus(i) = x_plus(i) + h;  % 在第 i 个维度上加上步长 h
            % 
            %     x_minus = x;
            %     x_minus(i) = x_minus(i) - h;  % 在第 i 个维度上减去步长 h
            % 
            %     % 使用中心差分公式计算梯度
            %     grad(i) = (obj.predict_proba(x_plus) - obj.predict_proba(x_minus)) / (2 * h);
            % end
            
        end
        




        % Hessian_B方法
        function hessian = Hessian_B(obj, x)
            x = x';
            % x
            % if length(x)==4
            %     x = x([1, 2])';
            % 
            % else
            %     x = x(:, 1:2);
            % end
            % x

            %这里计算有问题，只用到了第一组的节点，而没有用到10组的节点
            %现在没问题了，已经改过来了
            W0 = obj.W(1:obj.dimension, :);
            W1 = obj.W(obj.dimension + 1:end, :);

            % Wh_matrix = obj.Wh{1};
            Wh_matrix = cell2mat(obj.Wh);



            K1 = (Wh_matrix(:, 1) ./ (obj.std + 1e-6)');
            K2 = (Wh_matrix(:, 1) ./ (obj.std + 1e-6)');
            gradient = 2 * (W1(1,:) * [0;1] * obj.grad2(((x - obj.mean) ./ (obj.std + 1e-6)) * Wh_matrix(:, 1) + obj.bh{1})) * (K1 * K2');



            for i = 2:size(Wh_matrix, 2)
                K1 = (Wh_matrix(:, i) ./ (obj.std + 1e-6)');
                K2 = (Wh_matrix(:, i) ./ (obj.std + 1e-6)');
                gradient = gradient + 2 * (W1(i,:) * [0;1] * obj.grad2(((x - obj.mean) ./ (obj.std + 1e-6)) * Wh_matrix(:, i) + obj.bh{ceil(i / obj.N2)})) * (K1 * K2');
            end
            hessian = gradient;

            % h = 1e-5;
            % 
            % assert(length(x) == 2, 'x应为一个2x1向量');
            % 
            % H = zeros(2, 2);
            % 
            % for i = 1:2
            %     for j = 1:2
            %         if i == j
            %             % 二阶导数（对角线）
            %             x_plus = x;
            %             x_plus(i) = x_plus(i) + h;
            %             x_minus = x;
            %             x_minus(i) = x_minus(i) - h;
            % 
            %             H(i, j) = (obj.predict_proba(x_plus) - 2*obj.predict_proba(x) + obj.predict_proba(x_minus)) / h^2;
            %         else
            %             % 混合二阶导数（非对角线）
            %             % 生成四个点：i±h, j±h
            %             x_pp = x;
            %             x_pp([i j]) = x_pp([i j]) + h;
            % 
            %             x_pm = x;
            %             x_pm(i) = x_pm(i) + h;
            %             x_pm(j) = x_pm(j) - h;
            % 
            %             x_mp = x;
            %             x_mp(i) = x_mp(i) - h;
            %             x_mp(j) = x_mp(j) + h;
            % 
            %             x_mm = x;
            %             x_mm([i j]) = x_mm([i j]) - h;
            % 
            %             % 计算四个点的函数值
            %             f_pp = obj.predict_proba(x_pp);
            %             f_pm = obj.predict_proba(x_pm);
            %             f_mp = obj.predict_proba(x_mp);
            %             f_mm = obj.predict_proba(x_mm);
            % 
            %             H(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4 * h^2);
            %         end
            %     end
            % end
            % hessian = H;
        end
        
        % LfB方法
        function result = LfB(obj, z)


            theta1 = z(1);
            theta2 = z(2);
            w1 = z(3);
            w2 = z(4);
            x = obj.L1 * cos(theta1) + obj.L2 * cos(theta2);
            y = obj.L1 * sin(theta1) + obj.L2 * sin(theta2);

            grad_B_x_y = obj.gradient_B(z);
            dB_dx = grad_B_x_y(1);
            dB_dy = grad_B_x_y(2);
            dB_dtheta1 = dB_dx*(-obj.L1*sin(theta1)-obj.L2*sin(theta1+theta2)) + dB_dy*(obj.L1*cos(theta1)+obj.L2*cos(theta1+theta2));
            dB_dtheta2 = dB_dx*(-obj.L2*sin(theta1+theta2)) + dB_dy*(obj.L2*cos(theta1+theta2));

            grad_B_theta1_theta2 = [dB_dtheta1;dB_dtheta2];
           
            result = grad_B_theta1_theta2' * [w1; w2];
        end
        
        function result = d_LfB(obj, z)

            theta1 = z(1);
            theta2 = z(2);
            w1 = z(3);
            w2 = z(4);
            x = obj.L1 * cos(theta1) + obj.L2 * cos(theta2);
            y = obj.L1 * sin(theta1) + obj.L2 * sin(theta2);

            grad_B_x_y = obj.gradient_B(z);
            Fx = grad_B_x_y(1);
            Fy = grad_B_x_y(2);
            dB_dtheta1 = Fx*(-obj.L1*sin(theta1)-obj.L2*sin(theta1+theta2)) + Fy*(obj.L1*cos(theta1)+obj.L2*cos(theta1+theta2));
            dB_dtheta2 = Fx*(-obj.L2*sin(theta1+theta2)) + Fy*(obj.L2*cos(theta1+theta2));

            Hessian_result = obj.Hessian_B(z);
            Fxx = Hessian_result(1,1);
            Fxy = Hessian_result(1,2);
            Fyy = Hessian_result(2,2);

            dx_dtheta1 = -obj.L1*sin(theta1)-obj.L2*sin(theta1+theta2);
            ddx_dtheta1 = -obj.L1*cos(theta1)-obj.L2*cos(theta1+theta2);
            dx_dtheta2 = -obj.L2 * sin(theta1+theta2);
            ddx_dtheta2 = -obj.L2 * cos(theta1+theta2);
            dy_dtheta1 = obj.L1*cos(theta1)+obj.L2*cos(theta1+theta2);
            ddy_dtheta1 = -obj.L1*sin(theta1)-obj.L2*sin(theta1+theta2);
            dy_dtheta2 = obj.L2*cos(theta1+theta2);
            ddy_dtheta2 = -obj.L2*sin(theta1+theta2);

            dFx_dtheta1 = Fxx*dx_dtheta1+Fxy*dy_dtheta1;
            dFx_dtheta2 = Fxx*dx_dtheta2+Fxy*dy_dtheta2;
            dFy_dtheta1 = Fxy*dx_dtheta1+Fyy*dy_dtheta1;
            dFy_dtheta2 = Fxy*dx_dtheta2+Fyy*dy_dtheta2;

            dLfB_dtheta1 = w1*dFx_dtheta1*dx_dtheta1+w1*Fx*ddx_dtheta1+w1*dFy_dtheta1*dy_dtheta1+w1*Fy*ddy_dtheta1...
                +w2*dFx_dtheta1*dx_dtheta2+w2*Fx*ddx_dtheta2+w2*dFy_dtheta1*dy_dtheta2+w2*Fy*ddy_dtheta2;

            dLfB_dtheta2 = w1*dFx_dtheta2*dx_dtheta1+w1*Fx*ddx_dtheta2+w1*dFy_dtheta2*dy_dtheta1+w1*Fy*ddy_dtheta2...
                +w2*dFx_dtheta2*dx_dtheta2+w2*Fx*ddx_dtheta2+w2*dFy_dtheta2*dy_dtheta2+w2*Fy*ddy_dtheta2;

            dLfB_dw1 = dB_dtheta1;
            dLfB_dw2 = dB_dtheta2;

            result = [dLfB_dtheta1;dLfB_dtheta2;dLfB_dw1;dLfB_dw2];

            % result = [v*cos(theta)*db2_dx2+v*sin(theta)*db2_dxdy, v*cos(theta)*db2_dxdy+v*sin(theta)*db2_dy2, gradient_B * [cos(theta);sin(theta)], gradient_B*[-v*sin(theta); v*cos(theta)]];      
        end

        % Lf2_B方法
        function result = Lf2_B(obj, z)
            theta1 = z(1);
            theta2 = z(2);
            w1 = z(3);
            w2 = z(4);
            x = obj.L1 * cos(theta1) + obj.L2 * cos(theta2);
            y = obj.L1 * sin(theta1) + obj.L2 * sin(theta2);

            result = obj.d_LfB(z)' * [w1; w2; 0; 0];

        end
        
        % LgLf_B方法
        function result = LgLfB(obj, z)
            theta1 = z(1);
            theta2 = z(2);
            w1 = z(3);
            w2 = z(4);
            x = obj.L1 * cos(theta1) + obj.L2 * cos(theta2);
            y = obj.L1 * sin(theta1) + obj.L2 * sin(theta2);
            
            B = [ 
                0, 0;
                0, 0;
                1, 0;
                0, 1
            ];
            result = obj.d_LfB(z)' * B;
        end
        
        function result = dB_dt(obj, x)
            result = obj.LfB(x);
        end



        % transform方法
        function inputdata = transform(obj, data)
            enhencedata = obj.enhence_generator.transform(data);
            inputdata = [data, enhencedata];
        end
        
        % partial_fit方法
        function obj = partial_fit(obj, extratraindata, extratrainlabel)

            xdata = obj.normalscaler.transform(extratraindata);
            xdata = obj.transform(xdata);

            label = zeros(size(extratrainlabel, 1), 2);
            for i=1:size(extratrainlabel, 1)
                if extratrainlabel(i)==-1
                    label(i,1) = 1;
                else if extratrainlabel(i)==1
                    label(i, 2) = 1;
                end
                end
            end
            
            delta_K = obj.reg2 * obj.K * xdata' * pinv(obj.reg2 * xdata * obj.K * xdata' + eye(size(obj.reg2 * xdata * obj.K * xdata', 1))) * xdata * obj.K;
            delta_Q = obj.reg2 * xdata' * label - xdata' * label * obj.C;

            obj.K = obj.K - delta_K;
            obj.Q = obj.Q + delta_Q;
            obj.W = obj.K * obj.Q;
        end
    end
end
