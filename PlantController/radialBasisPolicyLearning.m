clear
close all 

% load trajectory data
load('xtraj2.mat');
load('utraj2.mat');

% Extract trajectories and control inputs
xtraj = xtraj2.eval(xtraj2.tt);
utraj = utraj2;

% redefine state variables
x = xtraj(:,1:1:2001);
T = length(x(1,:));
t = 1:T;
u = utraj(:,1:1:2001);
c = x;
M = length(x(1,:));
a = ones(1,M);

% Get radial basis function approximation
policyRBF = PolicyGradientFA(0, 0, 0, 0, 4, t, length(t));

% Fit data to radial basis - this function updates the FA weights
policyRBF = policyRBF.fitFA(x, u, t);
policyRBF = policyRBF.approximate(t);
u_est = policyRBF.approximator; 

% Approximation error
error = (u - u_est);
rbfFA_error = norm(error, 2);
fprintf("\nRadial Basis Function Approximation Error: %2.5f\n", rbfFA_error); 

% Overlay approximations with true control data
f = figure();
for i = 1:4
    plot(u_est(i,:), '.');
    hold on
    plot(u(i,:));
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

% Set up plant
options = [];
options.floating = true;
options.terrain = RigidBodyFlatTerrain();
options.use_bullet = false;

m = PlanarRigidBodyManipulator('KneedCompassGait_noankles.urdf', options);
r = TimeSteppingRigidBodyManipulator(m,.001);

% Setup visualizer - comment out for speed
% v = r.constructVisualizer;
% v.axis = [-1.7 5.2 -0.1 1.6];
% v.display_dt = .05;

% Set up global variables
global state_targets;
global current_target_state;
global last_update_time;

current_target_state = 3; % starting state - can only be 1 or 3 for realistic starting poses
last_update_time = 0;

torso_lean = 0.025;
max_hip_angle = 1.4;
max_knee_angle = 0.6;
leg_cross = 1.2;
straight_knee = 0.1;
bend_ankle = pi/2 + 0.1;
kick_ankle = pi/2;

% left leg, left knee, left ankle, right leg, right knee, right ankle
state_targets = {
        [-leg_cross/2 - torso_lean; max_knee_angle; bend_ankle; 0; straight_knee; bend_ankle],...% left bend
        [max_hip_angle/2 - torso_lean; straight_knee; kick_ankle; 0; straight_knee; kick_ankle],... % left kick back
        [0; straight_knee; bend_ankle; -leg_cross/2 - torso_lean; max_knee_angle; bend_ankle],... % right bend
        [0; straight_knee; kick_ankle; max_hip_angle/2 - torso_lean; straight_knee; kick_ankle],... % right kick back
      };

explorationFactor = 10;
sigma = explorationFactor*eye(length(u(:,1)));
% 
% % Evaluate fisher information matrix 
% F = policyRBF.fisherInformation(policyRBF.linearFA{1}.phi, sigma);
% F = F + eye(length(F(1,:))); % ensure F is well-conditioned


%% Run RL 

nepisodes = 20;

% Initialize basis functions
for k = 1:T
    phi(:,:,k) = [policyRBF.linearFA{1}.phi(k,:), zeros(1,3*policyRBF.linearFA{1}.M);
               zeros(1,policyRBF.linearFA{2}.M), policyRBF.linearFA{2}.phi(k,:), zeros(1,2*policyRBF.linearFA{2}.M);
               zeros(1,2*policyRBF.linearFA{3}.M), policyRBF.linearFA{3}.phi(k,:), zeros(1,policyRBF.linearFA{3}.M);
               zeros(1,3*policyRBF.linearFA{4}.M), policyRBF.linearFA{4}.phi(k,:)];
end

baseline = zeros(1,T);

for k = 1:nepisodes

    % Estimate policy gradient
    [policyGradient, baseline] = policyRBF.policyGradient(sigma, u_est, r, phi, baseline, T);

    % Update policy based on policy gradient estimate
    policyRBF = policyRBF.updatePolicy(policyGradient);

    % Determine which sequence of states we should use to evaluate the policy.
    % This sequence changes the value of the basis functions at each time step 
    controls = policyRBF.approximate(t);
    
    % Plot approximator and previous control input
    f = figure(1);
    plot(u_est')
    hold on
    plot(controls.approximator', '.')
    plot(u_est' - controls.approximator')
    saveas(f, 'policy10.png');
    hold off

    % Display norm of difference between current approximator and previous
    % controller estimate
    fprintf('Trajectory Deviation: %d\n', norm(u_est' - controls.approximator'));

    % update expected control input based on updated policy
    u_est = controls.approximator;
end