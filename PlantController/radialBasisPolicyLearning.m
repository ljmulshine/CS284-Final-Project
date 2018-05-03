clear
close all 
% 
% % load trajectory data
% load('xtraj2.mat');
% load('utraj2.mat');
% 
 
T = 2001; 
states = 1:14;
% % Extract trajectories and control inputs
% xtraj = xtraj2.eval(xtraj2.tt);
% xtraj = xtraj(:,1:T);
% u = utraj2(:,1:T);

load('good_xtrajs.mat');
load('good_utrajs.mat');


good_xtrajs = good_xtrajs(:,:,150); 
good_utrajs = good_utrajs(:,:,150); 

num_trajs = length(good_utrajs(1,1,:));

xtraj = sum(good_xtrajs, 3) ./ num_trajs; 
utraj = sum(good_utrajs, 3) ./ num_trajs;

xtraj_training = reshape(good_xtrajs, 14, 2001 * num_trajs);
utraj_training = reshape(good_utrajs, 4, 2001 * num_trajs);

u_diff = diff(utraj');
figure()
plot(u_diff);


%  Define indices of centers used for RBF approximation
t = [1:4:50, 51:10:110, ... 
     111:4:170, 171:10:200,...
     201:4:225, 226:10:475,...
     476:4:650, 651:10:910, ...
     911:4:1000, 1001:10:1320, ...
     1321:4:1360, 1361:10:1415, ...
     1416:4:1440, 1441:10:1811, ...
     1812:4:1930, 1931:10:2001 ];

% Define the centers used for RBF approximation
centers = xtraj(states,t);

% Plot centers (overlay centers with original trajectory)
figure()
plot(t, centers');
hold on
plot(1:T, xtraj(states,:)');

% Number of basis functions
nbasis_functions = length(t);
% Number of actuators/actions
nactions = 4;

% Get radial basis function approximation
policyRBF = PolicyGradientFA(nactions, centers, nbasis_functions);

% Fit radial basis functions to initial trajectory
policyRBF = policyRBF.fitFA(xtraj_training(states,:), utraj_training);

for i = 1:T
    u_est(:,i) = policyRBF.evaluate(xtraj(states,i));
end

figure()
plot(u_est');

% Approximation error
error = (utraj - u_est);
rbfFA_error = norm(error, 2);
fprintf("\nRadial Basis Function Approximation Error: %2.5f\n", rbfFA_error); 

% Overlay approximations with true control data
f = figure();
for i = 1:4
    plot(u_est(i,:), '.');
    hold on
    plot(utraj(i,:));
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

% 
% % Evaluate fisher information matrix 
% F = policyRBF.fisherInformation(policyRBF.linearFA{1}.phi, sigma);
% F = F + eye(length(F(1,:))); % ensure F is well-conditioned

alpha = 0.90;
nepisodes = 500;
reward = zeros(1,nepisodes);
baseline = zeros(1,T);

%% Run RL 

global prev_action_norm;

prev_action_norm = zeros(1,T);

for i = 1:T
    prev_action_norm(i) = norm(u_est(:,i));
end
for k = 1:nepisodes
    
    alpha = alpha - 0.5*1/nepisodes;
    
    % Estimate policy gradient
    [policyGradient, baseline, reward(k)] = policyRBF.policyGradient(sigma, u_est, r, baseline, T, alpha);

    % Update policy based on policy gradient estimate
    policyRBF = policyRBF.updatePolicy(policyGradient);

    % Plot approximator and previous control input
    f = figure(1);
    plot(u_est')
    for i = 1:T
        u_est_new(:,i) = policyRBF.evaluate(xtraj(states,i));
    end
    hold on 
    plot(u_est_new');
   
    % Display norm of difference between current approximator and previous
    % controller estimate
    fprintf('Trajectory Deviation: %d\n', norm(u_est' - u_est_new'));

    % update expected control input based on updated policy
    u_est = u_est_new;
end