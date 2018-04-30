clear
close all 

load('trajectoryData.mat');

upper_left_leg_pin_state = [xtraj(1,:); xtraj(2,:); xtraj(3,:); xtraj(4,:); xtraj(8,:); xtraj(9,:); xtraj(10,:); xtraj(11,:)];

% state space parameters
state_ub = max(upper_left_leg_pin_state,[],2)';
state_lb = min(upper_left_leg_pin_state,[],2)';

% fourier basis order
fa_order = 3;

% construct fourier basis
%
% at this point, the fourier basis functions (features) are defined and can be
% evaluated at state s by calling fourierBasis.computeFeatures(s). 
policyFA = PolicyGradientFA(length(state_ub), state_lb, state_ub, fa_order, 1);

%% Initial Trajectory Data

% time interval
t = t(1:100:end/4);
% states
x = upper_left_leg_pin_state(:,1:100:end/4);
% control inputs
u = utraj(1,1:100:end/4);

% plot test data
figure()
subplot(2,1,1);
plot(xtraj(1,:));
hold on
plot(xtraj(2,:));
plot(xtraj(3,:));
plot(xtraj(4,:));
subplot(2,1,2);
plot(utraj(1,:));


%% Learn weights to approximate control output using Fourier Basis

% fit data to fourier basis - this function updates the FA weights
policyFA = policyFA.fitFA(x, u, t);

% find approximate control inputs along trajectory x using fourier basis 
% with updated weights
u_est = policyFA.approximate(x).approximator;

% plot approximate function
figure()
plot(t, u_est);

error = norm(u - u_est);

%% Policy Gradients
sigma = 0.1*eye(length(u(1,:)));
w = policyFA.linearFA{1}.weights;
psi = policyFA.linearFA{1}.getBasisFunctions(x);

% define arbitrary trajectory
traj = x;

% evaluate policy gradient
policyGradient = policyFA.policyGradient(psi, sigma, u_est, w, traj);

% evaluate fisher information matrix 
F = policyFA.fisherInformation(psi, sigma);

% ensure Fisher infomation matrix is not ill-conditioned
F = F + eye(length(F(1,:)));

% update policy based on policy gradient
delta = 1; % the magnitude of delta controls how much variation we allow between policy iterations
policyFA = policyFA.updatePolicy(policyGradient, F, delta);

% determine which sequence of states we should use to evaluate the policy.
% This sequence changes the value of the basis functions at each time step 
trajectory = policyFA.approximate(traj).approximator;

figure()
plot(u_est)
hold on
plot(trajectory)
plot(u_est - trajectory)
legend('Initial Policy', 'Updated Policy', 'Deviation'); 

