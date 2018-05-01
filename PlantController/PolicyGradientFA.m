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
                features = obj.linearFA{i}.getBasisFunctions(t);
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
                
                if abs(x(2,i) - 1) < 0.003
                    reward(i) = reward(i) + 100;
                end
                if abs(x(2,i) - 1) < 0.1
                    reward(i) = reward(i) + 10;
                end
                if abs(x(2,i) - 1) < 0.2
                    reward(i) = reward(i) + 1;
                end
                if abs(x(2,i) - 1) < 0.25
                    reward(i) = reward(i) + 0.5;
                end
                
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
        function [g, baseline] = policyGradient(obj, sigma, u_est, r, phi, baseline, T)

            % number of policy samples
            N = 10;
            
            % initialize policy gradient
            g = zeros(4*obj.linearFA{1}.M,1);
            
            
            % inverse covariance
            iS = inv(sigma);
            
            % allocate space for trajectory and policies
            sampled_policy = zeros(4,T,N);
            sampled_traj = zeros(14,T,N);           
              
            % construct PD controller - note that with the current set up, these
            % constants are not used and thus have no effect
            Kp = 170; % 170 is a good value for just FB
            Kd = 2*sqrt(Kp);
            cv = 0.1; % 0.1 is a good value for just FB
            cd = 0;
            alpha = 0.9;
            
            % Iterate over a number of policy samples to more accurately
            % estimate policy gradient
            for i = 1:N
                % construct sample policy
                for t = 1:T
                    sampled_policy(:,t,i) = obj.samplePolicy(u_est(:,t), sigma)';
                end
                u_sampled = sampled_policy(:,:,i);
                
                % Simulate policy and get trajectory
                xtraj = runPDx(Kp,Kd,cv,cd,r,alpha,u_sampled);
                sampled_traj(:,:,i) = xtraj.eval(linspace(0,2,T));
                traj = sampled_traj(:,:,i);
                
                if (min(traj(2,:)) > 0.95)

                    figure(2)
                    plot(traj(2,:));
                    title('Base Height');

                    % Evaluate "return" from sampled trajectory
                    R(:,i) = obj.R_t(traj);

                    % Evaluate advantage function at each point in time
                    A = R(:,i) - baseline;

                    % Calculate policy gradient at each time step
                    for t = 1:T

                        w = [obj.linearFA{1}.weights; 
                             obj.linearFA{2}.weights; 
                             obj.linearFA{3}.weights; 
                             obj.linearFA{4}.weights];

                        % evaluate policy gradient
                        g(:,t, i) = phi(:,:,t)' * iS * (u_sampled(:,t) - phi(:,:,t) * w).*A(t);
                    end
                else
                    g(:,t, i) = zeros(1,4 * obj.linearFA{1}.M);
                end
                    
            end
          
            % Evaluate new baseline reward
            baseline = sum(R,2) / N;
            
            % Average over N samples
            g = (1/(N*T))*sum(sum(g,2),3);
        end
        
        % Return fisher information matrix for parameterized stochastic
        % policy, pi
        function F = fisherInformation(obj, psi, sigma)
            F = psi'* inv(sigma) * psi;
        end
        
        % Update the policy based on policy gradient in current iteration
        function obj = updatePolicy(obj, g)
            alpha = 0.5;
            % update FA parameters based on policy gradient
            del = reshape(alpha * g, 2001, 4);
            for i = 1:obj.nactions
                obj.linearFA{i}.weights = obj.linearFA{i}.weights + del(:,i);
            end
        end
        
    end
end

