function [xtraj, ufb] = runPDx(Kp,Kd,cv,cd,r,alpha, utraj, sigma)
    
    % This function simulates KneedCompassGait model using a SIMBICON-style
    % controller with alpha-weighted feedback.
    % @param utraj    feedforward function approximator handle
    % @param sigma    covariance matrix between control actuator torques
    % @param alpha    feedback weight 
    
    % Declare global variables
    global state_targets;
    global current_target_state;
    global last_update_time;

    % Set initial condition
    x0 = Point(r.getStateFrame());
    x0.base_z = 1;
    x0.left_upper_leg_pin = state_targets{current_target_state}(1);
    x0.right_upper_leg_pin = state_targets{current_target_state}(4);
    x0.left_knee_pin = state_targets{current_target_state}(2);
    x0.right_knee_pin = state_targets{current_target_state}(5);

    % Update state
    current_target_state = mod(current_target_state + 1,4);
    if current_target_state == 0
        current_target_state = 4;
    end

    % construct PD controller
    c = PDController(r,Kp,Kd,cv,cd,alpha,utraj, sigma);
    
    % Build feedback system
    sys = feedback(r,c);

    % Run simulation, then play it back at realtime speed
    xtraj = simulate(sys,[0 2],double(x0));

    % Find controller feedback outputs corresonding to each time step - 
    % - used to generate traj for warm starting RL
    
    % Reset the target state to ensure that the right target poses are selected
    
    current_target_state = 4;
    
    ufb = zeros(4,length(xtraj.tt));    
    
    for i = 1:length(xtraj.tt)
        x = xtraj.xx(:,i);
        ufb(:,i) = c.getUfb(xtraj.tt(i),0,x);
    end

end