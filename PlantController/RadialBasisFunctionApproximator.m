
classdef RadialBasisFunctionApproximator
 % This class serves as the framework for radial basis function
 % approximation
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
    function obj = RadialBasisFunctionApproximator(nactions, c, M)
        obj.centers = c;
        obj.nstate_vars = nactions;
        obj.M = M;
        obj.ncenters = length(obj.centers(1,:));
        obj.width = repmat(20, 1, obj.ncenters);
    end
    
    % Compute the feature vector for a given state using Gaussian Radial
    % Basis functions
    %
    % @param s		the state in question.
    % @return phi	a vector of doubles representing each basis function evaluated at s
    function features = computeFeatures(obj, s)
        a = repmat(s,1,obj.ncenters) - obj.centers;
        r2 = sum(a.*a, 1);
        features = exp(- r2 ./ (obj.width.^2));
    end
    
    % Compute feature vectors for each state along trajectory x.
    function psi = getBasisFunctions(obj, x)
        % number of time points
        T = length(x(1,:));
        
        % compute feature vector for each time point 
        psi = zeros(T, obj.M);
        for i = 1:T
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

  end
end

