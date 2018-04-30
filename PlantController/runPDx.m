% Setup KneedCompassGait plant
tic
% function zf = runPDx(Kp,Kd,cv,cd,r)
options = [];
options.floating = true;
options.terrain = RigidBodyFlatTerrain();
options.use_bullet = false;

m = PlanarRigidBodyManipulator('KneedCompassGait_noankles.urdf', options);
r = TimeSteppingRigidBodyManipulator(m,.001);

% Setup visualizer
% v = r.constructVisualizer;
% v.axis = [-1.7 5.2 -0.1 1.6];
% v.display_dt = .05;

% Set up global variables
% global state;
global state_targets;
global current_target_state;
global stance_time;
global last_update_time;

current_target_state = 1; % starting state
stance_time = 0;
last_update_time = 0;

torso_lean = 0.1;
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

% Set initial condition
x0 = Point(r.getStateFrame());
x0.base_z = 1;
x0.left_upper_leg_pin = state_targets{current_target_state}(1);
x0.right_upper_leg_pin = state_targets{current_target_state}(4);
x0.left_knee_pin = state_targets{current_target_state}(2);
x0.right_knee_pin = state_targets{current_target_state}(5);

% Update state
current_target_state = mod(current_target_state + 1,4);
% if current_target_state == 0
%     current_target_state = 4;
% end

% construct PD controller
Kp = 170;
Kd = 2*sqrt(Kp);
cv = 1;
cd = 5;
% c = PDController(r,Kp,Kd,cv,cd);
c = PDController(r,Kp,Kd,cv,cd,xtraj1,utraj1);

sys = feedback(r,c);

% Run simulation, then play it back at realtime speed
xtraj = simulate(sys,[0 3],double(x0));
% playback(v,xtraj,struct('slider',true));

% Find controller outputs corresonding to each time step
% u = zeros(4,length(xtraj.tt));
% for i = 1:length(xtraj.tt)
%     x = xtraj.xx(:,i);
%     u(:,i) = c.output(xtraj.tt(i),0,x);
% end

zf = xtraj.xx(2,end);
% end
toc