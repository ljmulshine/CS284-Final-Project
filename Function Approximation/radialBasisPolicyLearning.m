clear
close all 

% load trajectory data
load('trajectoryData.mat');

% redefine state variables
x = xtraj(:,2:100:end);
x = 1:length(x);
u = utraj(:,2:100:end);
c = x;
M = length(x(1,:));
a = ones(1,M);

% Get radial basis function approximation
policyRBF = PolicyGradientFA(length(x(:,1)), 0, 0, 0, 4, x, length(x(1,:)));

% Fit data to radial basis - this function updates the FA weights
policyRBF = policyRBF.fitFA(x, u, t);
policyRBF = policyRBF.approximate(x);
u_est = policyRBF.approximator; 

% Approximation error
error = (u - u_est);
rbfFA_error = norm(error, 2);
fprintf("\nRadial Basis Function Approximation Error: %2.5f\n", rbfFA_error); 

% Overlay approximations with true control data
f = figure();
for i = 1:4
    plot(u_est(i,:));
    hold on
    plot(u(i,:), '.');
end
title('Initial Policy Estimate');
xlabel('Time Step');
ylabel('Control Input');
legend('upper-left-leg-pin', 'left-knee-pin', 'upper-right-leg-pin', 'right-knee-pin');
saveas(f, 'policy.png');

% Plot error for each actuator
figure()
subplot(4,1,1);
plot(error(1,:));
title('Error (upper-left-leg-pin)');
subplot(4,1,2);
plot(error(2,:));
title('Error (left-knee-pin)');
subplot(4,1,3);
plot(error(3,:));
title('Error (upper-right-leg-pin)');
subplot(4,1,4);
plot(error(4,:));
title('Error (right-knee-pin)');

%% Policy Gradients
explorationFactor = 100;
sigma = explorationFactor*eye(length(u(1,:)));

% Evaluate fisher information matrix 
F = policyRBF.fisherInformation(policyRBF.linearFA{1}.phi, sigma);
F = F + eye(length(F(1,:))); % ensure F is well-conditioned

for j = 1:policyRBF.nactions
    % Get radial basis function weights for actuator control input function
    % approximator, j
    w = policyRBF.linearFA{j}.weights;
   
    % Estimate policy gradient
    policyGradient = policyRBF.policyGradient(policyRBF.linearFA{j}.phi, sigma, u_est(j,:), w);

    % Update policy based on policy gradient estimate
    delta = 10; % the magnitude of delta controls how much variation we allow between policy iterations
    policyRBF = policyRBF.updatePolicy(policyGradient, F, delta, j);
end

% Determine which sequence of states we should use to evaluate the policy.
% This sequence changes the value of the basis functions at each time step 
controls = policyRBF.approximate(x);

f = figure();
plot(u_est')
hold on
plot(controls.approximator')
plot(u_est' - controls.approximator')
legend('Initial Policy', 'Updated Policy', 'Deviation'); 
saveas(f, 'policy10.png');

fprintf('%d', norm(u_est' - controls.approximator'));

% update expected control input based on updated policy
u_est = controls.approximator;