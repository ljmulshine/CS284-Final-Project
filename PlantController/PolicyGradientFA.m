classdef PolicyGradientFA
    % Learn optimal policy through RL on function approximator using
    % policy gradietns
    
    properties
        linearFA
        approximator
        nactions
    end
    
    methods
        function obj = PolicyGradientFA(nactions, centers, M)
            obj.nactions = nactions;
            obj.linearFA = {};
            % Radial Basis Function Approximation
            for j=1:nactions
               obj.linearFA{j} = RadialBasisFunctionApproximator(nactions, centers, M); % copy one set of BFs for each discrete action
            end   
        end
            
        % Fit the function approximator to trajectory (x, u)
        function obj = fitFA(obj, x, u, t)
            % Compute basis features for each state along the trajectory
            for i = 1:obj.nactions
                features = obj.linearFA{i}.getBasisFunctions(x);
                obj.linearFA{i}.phi = features;
                
                % Compute optimal weights 
                w = lsqlin(obj.linearFA{i}.phi, u(i,:));

                % set weights
                obj.linearFA{i}.weights = w;
            end
        end
        
        
        % Evaluate function approximator at state x
        function u = evaluate(obj, s)
            for i = 1:obj.nactions
                u(i,1) = obj.linearFA{i}.computeFeatures(s) * obj.linearFA{i}.weights;
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
            
            global prev_action_norm
            
            reward = zeros(1,length(x(1,:)));
            for i = 1:length(x(1,:))
                reward(i) = reward(i) + -20000*(x(2,i)-1).^2 + 100;
                reward(i) = reward(i) + (prev_action_norm(i) - norm(a(:,i)));
                prev_action_norm(i) = norm(a(:,i));
            end
        end
        
        function R_t = R_t(obj, x, policy)
            % Evaluate reward at each point along trajectory of length N
            N = length(x(1,:));
            for i = 1:N
                % Calculate reward given current state and action
                reward(i) = obj.reward(x(:,i), policy(:,i));
            end

            gamma = 0.99;
            % Evaluate value function at each state along the trajectory
            for i = 1:N
                for j = (i + 1):N
                    rw(j-i) = gamma^(j - i - 1) * reward(j);
                end
                R_t(i) = sum(rw(1:(end - i + 1)));
            end
        end


        %  Evaluate the policy gradient of the given function approximator
        %  @param   psi     set of basis functions evaluate for all time, T
        %  @param   sigma   covariance of policy
        %  @param   u_est   most recent policy estimate
        %  @param   w       policy approximator weights
        %  @param   s       trajectory
        function [g, baseline, reward] = policyGradient(obj, sigma, u_est, r, baseline, T, alpha)
            
            global states
            global actions
            
            % Allocate space for states and actions along trajectory
            states = zeros(14, T);
            actions = zeros(4, T);
            
            % Number of policy samples
            N = 20;
            
            % Initialize policy gradient
            g = zeros(4*obj.linearFA{1}.M,1);
                      
            % Calculate inverse covariance
            iS = inv(sigma);
            
            % Allocate space for trajectory and policies
            sampled_policy = zeros(4,T,N);
            sampled_traj = zeros(14,T,N);           
              
            % construct PD controller - note that with the current set up, these
            % constants are not used and thus have no effect
            Kp = 170; % 170 is a good value for just FB
            Kd = 2*sqrt(Kp);
            cv = 0.1; % 0.1 is a good value for just FB
            cd = 0;                
            sigma = 1;
            
            % Iterate over a number of policy samples to more accurately
            % estimate policy gradient
            for i = 1:N
                % Simulate policy and get trajectory
                runPDx(Kp,Kd,cv,cd,r,alpha,@obj.evaluate, sigma);
               
                % Update states and actions for current policy sample
                sampled_traj(:,:,i) = states;
                sampled_policy(:,:,i) = actions;
                traj = states;
                policy = actions;
                
                figure(3) 
                plot(policy');
                
                % Plot base height over time - demonstrates policy
                figure(2)
                plot(traj(2,:));
                title('Base Height');

                % Evaluate "return" from sampled trajectory
                R(:,i) = obj.R_t(traj, policy);

                % Evaluate advantage function at each point in time
                A = R(:,i) - baseline;

                M = obj.linearFA{1}.M;
                % Calculate policy gradient at each time step
                for t = 1:T
                    phi = [ obj.linearFA{1}.computeFeatures(traj(:,t)), zeros(1,M), zeros(1,M), zeros(1,M);
                            zeros(1,M), obj.linearFA{2}.computeFeatures(traj(:,t)), zeros(1,M), zeros(1,M);
                            zeros(1,M), zeros(1,M), obj.linearFA{3}.computeFeatures(traj(:,t)), zeros(1,M);
                            zeros(1,M), zeros(1,M), zeros(1,M), obj.linearFA{4}.computeFeatures(traj(:,t)) ];

                    % Evaluate policy gradient
                    g(:,t, i) = phi' * iS * (actions(:,t) - obj.evaluate(traj(:,t))).*A(t);
                end               
            end
            
            % Evaluate new baseline reward
            beta = 0.5;
            baseline = beta * baseline + (1 - beta) * sum(R,2) / N;
            
            % Evaluate total reward
            reward = sum(R(1,:)) / N;
            
            fprintf("Reward: %f", reward);
            % Average over N samples
            g = (1/(N*T))*sum(sum(g,2),3);
        end
        
        % Update the policy based on policy gradient in current iteration
        function obj = updatePolicy(obj, g)
            alpha = 0.5;
            % update FA parameters based on policy gradient
            del = reshape(alpha * g, obj.linearFA{1}.M, 4);
            for i = 1:obj.nactions
                obj.linearFA{i}.weights = obj.linearFA{i}.weights + del(:,i);
            end
        end
        
    end
end

