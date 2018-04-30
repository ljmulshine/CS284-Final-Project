classdef PDController < DrakeSystem

  properties
    p
    Kp
    Kd
    cv
    cd
    qdes
    q_num
    qdot_num
    use_lqr % set to false as it is not linearizable
    in_rl % in training?
    in_pb % in playback mode?
    xtraj
    utraj
  end

  methods
    function obj = PDController(plant,Kp,Kd,cv,cd,xtraj,utraj)
%     function obj = PDController(plant,Kp,Kd,cv,cd)
        global current_target_state
        obj = obj@DrakeSystem(0,0,14,4,true,false);
        obj.p = plant;
%         obj.qdes = obj.getState(current_target_state);
        obj.use_lqr = 0; % use for testing lqr gains - dlinearize gives error that the system is not controllable
        [obj.Kp, obj.Kd] = obj.calcK(Kp,Kd);
        obj.cv = cv;
        obj.cd = cd;
        obj = obj.setInputFrame(plant.getStateFrame);
        obj = obj.setOutputFrame(plant.getInputFrame);
        obj.q_num = getNumPositions(obj.p);
        obj.qdot_num = getNumVelocities(obj.p);
        obj.in_rl = 0;
        obj.in_pb = 0;
        obj.xtraj = xtraj;
        obj.utraj = utraj;
        global state_targets;
    end
    
    function s = getState(obj,stateind,x)
        global state_targets
        s_new = obj.feedback_adjust_state(stateind, x);
        s = Point(getStateFrame(obj.p));
%         s.left_upper_leg_pin = state_targets{stateind}(1);
%         s.right_upper_leg_pin = state_targets{stateind}(4);
%         s.left_knee_pin = state_targets{stateind}(2);
%         s.right_knee_pin = state_targets{stateind}(5);
        s.left_upper_leg_pin = s_new(1);
        s.right_upper_leg_pin = s_new(4);
        s.left_knee_pin = s_new(2);
        s.right_knee_pin = s_new(5);
        s = double(s);
    end
    
    function y = updateState(obj,t,old_y,x)
      global last_update_time

      [left_h, left_x] = obj.left_foot_coords(x);
      [right_h, right_x] = obj.right_foot_coords(x);

      state = old_y(1);
      last_update_time = old_y(2);
      update_interval = 0.4;
      
      time_up = t - last_update_time > update_interval;
%       max_swing_dist = 0.8;
      if state == 1 || state == 3
          should_update = time_up; %| abs(left_x - right_x) > max_swing_dist;
      elseif state == 2
          should_update = (left_h < 0.0005);
      else
          should_update = (right_h < 0.0005);
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

    function u = output(obj,t,y,x)
        
        [left_h, left_x] = obj.left_foot_coords(x);
        left_contact = left_h < 0.005;
        [right_h, right_x] = obj.right_foot_coords(x);
        right_contact = right_h < 0.005;
        
%         global stance_time
        global current_target_state
        global last_update_time
        
%         if (current_target_state == 1 || current_target_state == 2)
%             if left_contact && (t-stance_time) > 0.1
%                 stance_time = t;
%             end
%         else
%             if right_contact && (t-stance_time) > 0.1
%                 stance_time = t;
%             end
%         end
        
        y = obj.updateState(t,[current_target_state; last_update_time],x);
        current_target_state = y(1);
        
        obj.qdes = obj.getState(current_target_state,x);
        
        
        u = -obj.Kp*(x(1:obj.q_num)-obj.qdes(1:obj.q_num))-obj.Kd*x(obj.q_num+1:obj.q_num+obj.qdot_num);
        
%         if obj.in_rl
%             u_rl = get_u();
%             u = u + u_rl;
%         end
        
        if obj.in_pb
            t_ind = find(obj.xtraj.tt == t);
            u = obj.utraj(:,t_ind);
        end
        
        u = min(u,200);
        u = max(u,-200);
        
    end
    
    function [Kp_full,Kd] = calcK(obj,Kp,Kd)
        if(obj.use_lqr)
            [A,B,C,D,xn0,y0] = obj.p.dlinearize(0.001,0,double(obj.qdes),0);
            Q = 1000*eye(size(A));
            R = eye(size(B,2));
            Kp = lqr(A,B,Q,R);
        else
            Kp_full = Kp*([ 0 0 0 1 0 0 0;...
                0 0 0 0 1 0 0;...
                0 0 0 0 0 1 0;...
                0 0 0 0 0 0 1]); % Only apply P control on positions and D control on velocities
            Kd = Kd/Kp*Kp_full;
        end
    end
    
    function [h, rel_x] = left_foot_coords(obj,x)
      base_z = x(2);
      base_relative_pitch = x(3);
      left_upper_leg_pin = x(4);
      left_knee_pin = x(5);


      [h, rel_x] = obj.foot_coords(base_z, left_upper_leg_pin + base_relative_pitch, left_knee_pin);
    end

    function h = left_foot_height(obj, x)
      [h, ~] = obj.left_foot_coords(x);
    end

    function [h, rel_x] = right_foot_coords(obj,x)

      base_z = x(2);
      base_relative_pitch = x(3);
      right_upper_leg_pin = x(6);
      right_knee_pin = x(7);

      [h, rel_x] = obj.foot_coords(base_z, base_relative_pitch + right_upper_leg_pin, right_knee_pin);
    end

    function h = right_foot_height(obj, x)
      [h, ~] = obj.right_foot_coords(x);
    end
    
    function [h, rel_x] = foot_coords(obj, base_z, hip_angle, knee_angle)
      h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle));
      rel_x = -0.5*(sin(hip_angle) + sin(hip_angle + knee_angle));
    end
    
    function state = feedback_adjust_state(obj, ind, x)
      global state_targets;
      v = x(8);
%       c_v = 0.2;
%       c_d = 2.2;

      state = state_targets{ind};
      if ind == 1
        state(1) = state(1) - obj.cv*v;
      elseif ind == 3
        state(4) = state(4) - obj.cv*v;
      elseif ind == 2
        [right_h, right_x] = obj.right_foot_coords(x);
        state(1) = state(1) - obj.cd * right_x;
      elseif ind == 4
        [left_h, left_x] = obj.left_foot_coords(x);
        state(4) = state(4) - obj.cd * left_x;
      end
    end
    
  end

end
