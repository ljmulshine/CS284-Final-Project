function xtraj = runPDx(Kp,Kd,cv,cd,r,alpha, utraj, sigma)

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
    % Kp = 170;
    % Kd = 2*sqrt(Kp);
    % cv = 0.1;
    % cd = 0;
    % c = PDController(r,Kp,Kd,cv,cd);
    c = PDController(r,Kp,Kd,cv,cd,alpha,utraj, sigma);

    sys = feedback(r,c);

    % Run simulation, then play it back at realtime speed
    xtraj = simulate(sys,[0 2],double(x0));

    % playback(v,xtraj,struct('slider',true)); % commented out for speed

    % % Find controller outputs corresonding to each time step - used to generate
    % % utraj for warm starting RL
    % current_target_state = 2;
    u = zeros(4,length(xtraj.tt));
    for i = 1:length(xtraj.tt)
        x = xtraj.xx(:,i);
        u(:,i) = c.output(xtraj.tt(i),0,x);
    end

    % zf = xtraj.xx(2,end); % for debugging

end