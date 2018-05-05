classdef PDController < DrakeSystem

  properties
    % plant
    p 
    
    % PD gains
    Kp
    Kd
    
    % State adjustment factors
    cv
    cd
    
    % Target state
    qdes
    
    % Number of positions and velocities in state
    q_num
    qdot_num
    
    alpha % relative weight between feedforward and feedback controllers
    ufb % controller feedback outputs
    
    utraj % controller outputs to be commanded (includes feedback and feedforward)
    sigma % covariance 
    
  end

  methods
    function obj = PDController(plant,Kp,Kd,cv,cd,alpha,utraj,sigma)
        % Initializes properties of the controller object        
        
        % Declare global variables
        global current_target_state
        global state_targets;
        
        % The kneed compass gait has a 14 dimensional state vector (number of inputs to controller)
        % 4 actuated joints (number of outputs to controller
        % The output is input-dependent and time-dependent
        obj = obj@DrakeSystem(0,0,14,4,true,false);
        
        obj.p = plant;
        
        % Set up controller gains
        [obj.Kp, obj.Kd] = obj.calcK(Kp,Kd);
        obj.cv = cv;
        obj.cd = cd;
        
        obj = obj.setInputFrame(plant.getStateFrame);
        obj = obj.setOutputFrame(plant.getInputFrame);
        
        % Set size of state space
        obj.q_num = getNumPositions(obj.p);
        obj.qdot_num = getNumVelocities(obj.p);

        % alpha = 1 --> only feedback; alpha = 0 --> only feedforward
        obj.alpha = alpha;
        obj.sigma = sigma;
        
        obj.ufb = [];
        
        obj.utraj = utraj;
    end
    
    function s = getState(obj,stateind,x)
        % Sets the target pose given the current state and mode
        % @param stateind current mode of system
        % @param x current state of system
        
        global state_targets
        
        % Adjust state to control for instability in the torso
        s_new = obj.feedback_adjust_state(stateind, x);
        
        s = Point(getStateFrame(obj.p));
        s.left_upper_leg_pin = s_new(1);
        s.left_knee_pin = s_new(2);
        s.right_upper_leg_pin = s_new(4);
        s.right_knee_pin = s_new(5);
        s = double(s);
    end
    
    function y = updateState(obj,t,old_y,x)
      % Checks to see if the mode needs to be updated and returns the
      % updated state and time of last update
      % @param t time of simulation
      % @param old_y 2-element vector with last state of system and time
      % since last update
      % @param x current state of system
      global last_update_time
      
      % Update depends on time since last update and heel strike detection

      state = old_y(1);
      last_update_time = old_y(2);
      update_interval = 0.4; % threshold for mode switching
      
      % check if enough time has elapsed
      time_up = t - last_update_time > update_interval;
      
      % calculate foot positions
      [left_h, ~] = obj.left_foot_coords(x);
      [right_h, ~] = obj.right_foot_coords(x);
      
      % check if time to update and if yes, return the new state and update
      % last_update_time
      if state == 1 || state == 3
          should_update = time_up;
      elseif state == 2
          should_update = (left_h < 0.005);
      else
          should_update = (right_h < 0.005);
      end
      
      if should_update
          state = mod(state+1, 4);
          
          if state == 0
              state = 4;
          end
          
          last_update_time = t;

      end
      y(1) = state;
      y(2) = last_update_time;
      
    end

    function u = output(obj,t,~,x)
        % Calculates the controller output
        % @param t Time of simulation
        % @param x Current state of plant
        
        % Declare global variables
        global current_target_state
        global last_update_time
        global states
        global actions
        
        % Update mode if necessary
        y = obj.updateState(t,[current_target_state; last_update_time],x);
        current_target_state = y(1);
        
        % Obtain target pose based on current mode
        obj.qdes = obj.getState(current_target_state,x);
        
        % Calculate PD feedback controller output
        % u_fb = -Kp(x-xd)-Kd*xdot
        u = -obj.Kp*(x(1:obj.q_num)-obj.qdes(1:obj.q_num))-obj.Kd*x(obj.q_num+1:obj.q_num+obj.qdot_num);
        
        % Determine time index
        t_ind = floor(t/obj.p.timestep+1);
        
        % Evaluate feedforward control inputs based on current state -
        % update global variables
        states(:,t_ind) = [reshape(x(1:obj.q_num), 7, 1); reshape(x(obj.q_num+1:obj.q_num+obj.qdot_num), 7, 1)];
        feedforward_controller = obj.utraj(states(:,t_ind));
        Sigma = obj.sigma^2*eye(4);
        actions(:,t_ind) = mvnrnd(feedforward_controller, Sigma);

        % Add in scaled feedforward inputs from utraj provided
        u = obj.alpha*u + (1-obj.alpha)*actions(:,t_ind);

        % Threshold controller outputs
        u = min(u,200);
        u = max(u,-200);
    end
    
    function ufb = getUfb(obj,t,~,x)
        % Calculates feedback commands to be called later for use in a
        % reward function that minimizes feedback torques. This is
        % equivalent to the output function without the feedforward
        % component, i.e., with alpha = 1
        
        % @param t Time of simulation
        % @param x Current state of plant
        
        % Declare global variables
        global current_target_state
        global last_update_time
        global actions
        
        % Update mode if necessary
        y = obj.updateState(t,[current_target_state; last_update_time],x);
        current_target_state = y(1);
        
        % Get current target state
        obj.qdes = obj.getState(current_target_state,x);
        
        % PD controller output
        ufb = -obj.Kp*(x(1:obj.q_num)-obj.qdes(1:obj.q_num))-obj.Kd*x(obj.q_num+1:obj.q_num+obj.qdot_num);

        % Threshold controller outputs
        ufb = min(ufb,200);
        ufb = max(ufb,-200);
    end
    
    function [Kp_full,Kd] = calcK(obj,Kp,Kd)
        % Calculates gain matrices to use in PD controller
        % @param Kp proportional controller gain
        % @param Kd derivative controller gain; generally defined as
        % 2*sqrt(Kp)
        
        Kp_full = Kp*([ 0 0 0 1 0 0 0;...
            0 0 0 0 1 0 0;...
            0 0 0 0 0 1 0;...
            0 0 0 0 0 0 1]); % Only apply P control on positions and D control on velocities
        
        Kd = Kd/Kp*Kp_full;
    end
    
    function [h, rel_x] = left_foot_coords(obj,x)
      % Calculates the left foot coordinates based on the current state
      base_z = x(2);
      base_relative_pitch = x(3);
      left_upper_leg_pin = x(4);
      left_knee_pin = x(5);
      [h, rel_x] = obj.foot_coords(base_z, left_upper_leg_pin + base_relative_pitch, left_knee_pin);
      
    end

    function [h, rel_x] = right_foot_coords(obj,x)
      % Calculates the right foot coordinates based on the current state
      base_z = x(2);
      base_relative_pitch = x(3);
      right_upper_leg_pin = x(6);
      right_knee_pin = x(7);

      [h, rel_x] = obj.foot_coords(base_z, base_relative_pitch + right_upper_leg_pin, right_knee_pin);
    end
    
    function [h, rel_x] = foot_coords(obj, base_z, hip_angle, knee_angle)
      % Calculates the foot coordinates based on the base height, and hip
      % and knee angles
      h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle));
      rel_x = -0.5*(sin(hip_angle) + sin(hip_angle + knee_angle));
    end
    
    function state = feedback_adjust_state(obj, ind, x)
      % Adjust target pose to account for torso instabilities
      % The update encourages the system to take a step forward or back if
      % the relation between the center of mass and foot position indicates 
      % instability
      % @param ind current mode that the system is in
      % @param x current state of system
      
      global state_targets;
      state = state_targets{ind};
      
      v = x(8); % base forward (x) velocity
      
      if ind == 1
        state(1) = state(1) - obj.cv*v;
        
      elseif ind == 3
        state(4) = state(4) - obj.cv*v;
        
      elseif ind == 2
        [~, right_x] = obj.right_foot_coords(x);
        state(1) = state(1) - obj.cd * right_x;
        
      elseif ind == 4
        [~, left_x] = obj.left_foot_coords(x);
        state(4) = state(4) - obj.cd * left_x;
        
      end
      
    end
    
  end

end
