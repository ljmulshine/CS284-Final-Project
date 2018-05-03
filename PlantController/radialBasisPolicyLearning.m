clear
close all 

% load trajectory data
load('xtraj2.mat');
load('utraj2.mat');


T = 2001; 
% Extract trajectories and control inputs
xtraj = xtraj2.eval(xtraj2.tt);
xtraj = xtraj(:,1:T);
u = utraj2(:,1:T);

u_diff = diff(u');

figure()
plot(u_diff);


%  Define indices of centers used for RBF approximation
t = [1:1:50, 51:1:110, ... 
     111:3:170, 171:3:200,...
     201:3:225, 226:3:475,...
     476:3:650, 651:3:910, ...
     911:3:1000, 1001:1:1320, ...
     1321:3:1360, 1361:1:1415, ...
     1416:2:1440, 1441:2:1811, ...
     1812:1:1930, 1931:1:2001 ];

% Define the centers used for RBF approximation
centers = xtraj(:,t);

% Plot centers (overlay centers with original trajectory)
figure()
plot(t, centers');
hold on
plot(1:T, xtraj');

% Number of basis functions
nbasis_functions = length(centers);
% Number of actuators/actions
nactions = 4;

% Get radial basis function approximation
policyRBF = PolicyGradientFA(nactions, centers, nbasis_functions);

% Fit radial basis functions to initial trajectory
policyRBF = policyRBF.fitFA(xtraj, u);

for i = 1:T
    u_est(:,i) = policyRBF.evaluate(xtraj(:,i));
end

figure()
plot(u_est');

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

explorationFactor = 10000;
sigma = explorationFactor*eye(length(u(:,1)));
% 
% % Evaluate fisher information matrix 
% F = policyRBF.fisherInformation(policyRBF.linearFA{1}.phi, sigma);
% F = F + eye(length(F(1,:))); % ensure F is well-conditioned


alpha = 0.99;
nepisodes = 500;
reward = zeros(1,nepisodes);

%% Run RL 

baseline = zeros(1,T);

for k = 1:nepisodes
    
    alpha = alpha - 0.5*1/nepisodes;

    % Estimate policy gradient
    [policyGradient, baseline, reward(k)] = policyRBF.policyGradient(sigma, u_est, r, baseline, T, alpha);

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
    hold off

    % Display norm of difference between current approximator and previous
    % controller estimate
    fprintf('Trajectory Deviation: %d\n', norm(u_est' - controls.approximator'));

    % update expected control input based on updated policy
    u_est = controls.approximator;
end