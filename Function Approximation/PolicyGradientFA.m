classdef PolicyGradientFA
    % Learn optimal policy through RL on function approximator using
    % policy gradietns
    
    properties
        linearFA
        approximator
        nactions
    end
    
    methods
        function obj = PolicyGradientFA(num_states, state_lb, state_ub, fa_order, nactions, c, M)
            obj.nactions = nactions;
            obj.linearFA = {};
            if (state_lb == 0)
                % Radial Basis Function Approximation
                for j=1:nactions
                   obj.linearFA{j} = RadialBasisFunctionApproximator(num_states, c, M); % copy one set of BFs for each discrete action
                end
            else
                % Fourier Basis Function Approximation
                for j=1:nactions
                  obj.linearFA{j} = FourierPolicyApproximator(num_states, state_lb, state_ub, fa_order); % copy one set of BFs for each discrete action
                end
            end    
            
        end
            
        % Fit the function approximator to trajectory (x, u)
        function obj = fitFA(obj, x, u, t)

            % Compute basis features for each state along the trajectory
            for i = 1:obj.nactions
                features = obj.linearFA{i}.getBasisFunctions(x);
                obj.linearFA{i}.phi = features;
                
                % Compute optimal weights 
                w = lsqlin(features, u(i,:));

                % set weights
                obj.linearFA{i}.weights = w;
            end

        end

        % Approximate function given input state vector (trajectory), x
        % 
        function obj = approximate(obj, x)
            % evaluate updated fourier basis approximator along trajectory to get
            % estimate of feedforward control inputs
            N = length(x(1,:));
            obj.approximator = zeros(obj.nactions, N);
            ClassName = class(obj);
            fprintf('%s', ClassName);
            if (strcmp(ClassName,'FourierPolicyApproximator'))
                for j = 1:obj.nactions
                    for i = 1:N
                        obj.approximator(j,i) = obj.linearFA{j}.valueAt(x(:,i));
                    end
                end
            else
                for j = 1:obj.nactions
                    obj.approximator(j,:) = obj.linearFA{j}.getBasisFunctions(x) * obj.linearFA{j}.weights;
                end
            end
        end
        

        % Sample from multivariate normal policy given mean, mu
        % and covariance, cov
        function traj = samplePolicy(obj, mu, sigma)
            % length of policy
            N = length(mu);
          
            % sample from stochastic policy
            traj = mvnrnd(mu, sigma)';
        end

        function reward = reward(obj, x, a)
            reward = zeros(1,length(x(1,:)));
            for i = 1:length(x(1,:))
                reward(i) = min(50, 1/abs(0.4 - x(8,i)));
                
                if abs(x(2,i) - 1) < 0.003
                    reward(i) = reward(i) + 100;
                end
                
                reward(i) = reward(i) - norm(a,2);
            end
        end
        
        
        function R_t = R_t(obj, x)
            % Evaluate reward at each point along trajectory of length N
            N = length(x(1,:));
            for i = 1:N
                % Get action at state s according to current policy
                actions = zeros(1,obj.nactions);
                for j = 1:obj.nactions
                    actions(j) = obj.linearFA{j}.valueAt(x(:,i));
                end
                
                % Calculate reward given current state and action
                reward(i) = obj.reward(x(:,i), actions);
            end

            % Evaluate value function at each state along the trajectory
            for i = 1:N
                R_t(i) = sum(reward(i:end));
            end
        end

        function Q = actionValueFunction(obj, s, a)
            % evaluate action value function (Q) at state s given action, a
            Q = 2;
        end

        function A = advantageFunction(obj, s, a)
            % evaluate advantage of taking action, a, in state, s
            A = obj.actionValueFunction(s,a) - obj.valueFunction(s);
        end

        %  Evaluate the policy gradient of the given function approximator
        %  @param   psi     set of basis functions evaluate for all time, T
        %  @param   sigma   covariance of policy
        %  @param   u_est   most recent policy estimate
        %  @param   w       policy approximator weights
        %  @param   s       trajectory
        function g = policyGradient(obj, psi, sigma, u_est, w)

            % number of policy samples
            N = 100;

            % number of time steps
            T = length(u_est);

            % initialize policy gradient
            g = zeros(T,1);
            alpha = 0.1;
            baseline = zeros(T,1);
            A = ones(T,1);
            B = psi' * inv(sigma);
            sampled_policy = zeros(T, N);
            sampled_traj = zeros(16, T, N);
            for i = 1:N
                % construct sample sample
                sampled_policy(:,i) = obj.samplePolicy(u_est, sigma);
                u_sampled = sampled_policy(:,i);
                
                % THIS IS WHERE I NEED TO CALL THE SIMULATOR
                % run sampled trajectory through simulator and get
                % trajectory 
                sampled_traj(:,:,i) = ones(16,400);
                traj = sampled_traj(:,:,i);
                
                % evaluate basis functions for sampled trajectory
                % psi = obj.linearFA{1}.getBasisFunctions(traj);
                
                % Evaluate "return" from sampled trajectory
                R(:,i) = obj.R_t(traj);
                
                % evaluate advantage function
                A = R(:,i) - baseline;

                % evaluate policy gradient
                g = g + (B * ((u_sampled - psi * w).*A));
            end

            baseline = sum(R,2) / N;
            
            % normalize policy gradient
            g = (1/(N*T))*g;
        end
        
        % Return fisher information matrix for parameterized stochastic
        % policy, pi
        function F = fisherInformation(obj, psi, sigma)
            F = psi'* inv(sigma) * psi;
        end
        
        % Update the policy based on policy gradient in current iteration
        function obj = updatePolicy(obj, g, F, delta, i)
            
            % update FA parameters based on policy gradient
            del = sqrt( delta / (g'* inv(F) * g)) * inv(F) * g
            obj.linearFA{i}.weights = obj.linearFA{i}.weights + del;
        end
        
    end
end

