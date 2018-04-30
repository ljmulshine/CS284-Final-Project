classdef FourierPolicyApproximator
 % Linear policy approximator using fourier basis
    
 properties
    weights
    phi
    alpha_scale
    multipliers
    nterms
    nstate_vars
    state_ub
    state_lb
    order
    approximator
  end

  methods
    % Construct a new FourierBasis with a given task specification and order. 
    % Assume a full basis initialized to 0 everywhere.
    %
    % @param num_states		
    % @param order the upper bound on individual FourierBasis coefficients.
    function obj = FourierPolicyApproximator(num_states, state_lb, state_ub, order)
      obj.order = order;
    	obj.nstate_vars = num_states;
      obj.nterms = (order + 1)^num_states;
    	obj.state_ub = state_ub;
    	obj.state_lb = state_lb;
      obj = obj.initialize(num_states, order);   
    end
    
    % Compute the feature vector for a given state.
    % This is achieved by evaluating each Fourier Basis function
    % at that state.
    %
    % @param s		the state in question.
    % @return phi		a vector of doubles representing each basis function evaluated at s
    function phi = computeFeatures(obj,s)
      phi = zeros(obj.nterms,1);
      for i=1:obj.nterms
        dsum = 0;
	
        for j=1:obj.nstate_vars
          sval = obj.scale(s(j), j);		
          dsum = dsum + sval * obj.multipliers(i,j);
        end        
    		phi(i) = cos(pi * dsum);
      end		
    end
    
    function psi = getBasisFunctions(obj, x)

        % get number of time steps in trajectory
        N = length(x(1,:));

        % compute basis features for each state along the trajectory
        psi = zeros(N, obj.nterms);
        for i = 1:N
            psi(i,:) = obj.computeFeatures(x(:,i));
        end

    end
	    
    % Scale a state variable to between 0 and 1 (this is required for the Fourier Basis)
    %
    % @param val	the state variable.
    % @param index	the state variable number.
    % @return		the normalized state variable
    function ss = scale(obj, val, index)
      ss = (val - obj.state_lb(index)) / (obj.state_ub(index) - obj.state_lb(index));
      if ss<0 || ss>1
        %disp(sprintf('index %d out of bounds: %f',index,val)); 
      end
    end
	
    % Initialize the FourierBasis using the given parameters.
    %
    % @param nvars		the number of state variables in the domain.
    % @param order		the basis order (the highest integer value of any individual coefficient). 
    function obj = initialize(obj, nvars, order) 
      % Obtain the coefficient vectors 
      obj = obj.computeFourierCoefficients(nvars, order);
      obj.nterms = length(obj.multipliers);

      % Create and initialize the feature and weight vectors.
      obj.phi = zeros(obj.nterms,1);
      obj.weights = zeros(obj.nterms,1);
      obj.alpha_scale = ones(obj.nterms,1);
    end

    % Compute the value of this function approximator at a given state.
    %
    % @param s		the state observation in question.
    % @return the value at state s.
    function v = valueAt(obj,s) 
      ph = obj.computeFeatures(s);
      v = obj.weights' * ph;
    end
    
    

    % Set the value of this function approximator everywhere.
    %
    % @param v		the target value.
    function obj = setValue(obj,v)
      % All weights are zero ...
      obj.weights = zeros(obj.nterms,1);
      obj.weights(1) = v;
    end
    
    % Add a weight delta vector to the function approximator's internal weights.
    %
    % @param w_delta	the vector to be added to the FA's weights
    function obj = addToWeights(obj, w_delta)
      obj.weights = obj.weights + obj.alpha_scale.*w_delta;
    end
    
    %
    % Set the function approximator's weight vector. 
    %
    % @param new_weights 	the new weight vector.
    function obj = setWeights(obj, new_weights)
      obj.weights = new_weights;
    end
    
    function obj = resetWeights(obj)
      obj = obj.setValue(0);
    end   

    function obj = computeFourierCoefficients(obj, nvars, order)
      obj.nterms = (order + 1)^nvars;
      obj.multipliers = zeros(obj.nterms,nvars);
    
      coeffs = zeros(1, nvars);
      cpos = 1;

      while (coeffs(nvars) <= order)
        obj.multipliers(cpos, :) = coeffs;
        cpos = cpos + 1;

        pos = 1;
        cont = 1;

        while (cont)
          coeffs(pos) = coeffs(pos) + 1;
          cont = 0;
          if(coeffs(pos) == order + 1)
            if(pos ~= nvars)
              coeffs(pos) = 0;
              pos = pos + 1;
              cont = 1;
            end
          end
        end
      end
    end
  end
   
end

