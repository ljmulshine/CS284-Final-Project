classdef RadialBasisFunctionApproximator
    
 properties
    weights
    phi
    centers
    ncenters
    nstate_vars
    M
    width
  end

  methods
    % Construct a new Radial Basis function approximator   
    % @param num_states		
    % @param M number of Radial Basis parameters used
    function obj = RadialBasisFunctionApproximator(num_states, c, M)
        obj.centers = c;
        obj.width = ones(1, M); % vector containing the widths of each basis function 
        obj.nstate_vars = num_states;
        obj.M = M;
        obj.ncenters = length(obj.centers(1,:));
    end
    
    % Compute the feature vector for a given state.
    %
    % @param s		the state in question.
    % @return phi		a vector of doubles representing each basis function evaluated at s
    function phi = computeFeatures(obj,s)
        a = repmat(s,1,obj.ncenters) - obj.centers;
        r2 = sum(a.*a, 1);
        phi = exp(- r2 ./ (obj.width.^2));
    end
    
    function psi = getBasisFunctions(obj, x)
        psi = zeros(obj.M, obj.M);
        for i = 1:length(x(1,:))
            psi(i,:) = obj.computeFeatures(x(:,i))';
        end
    end

    % Compute the value of this function approximator at a given state.
    %
    % @param s		the state observation in question.
    % @return the value at state s.
    function v = valueAt(obj,s) 
        v = obj.computeFeatures(s) * obj.weights;
    end
   
    % Set the value of this function approximator everywhere.
    %
    % @param v		the target value.
    function obj = setValue(obj,v)

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
   
  end

end

