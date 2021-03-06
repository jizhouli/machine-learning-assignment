function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    feature_num = size(X, 2);
    next_theta = zeros(size(theta));
    for itheta = 1:feature_num
        % update one theta each iteration of itheta
        sigma = 0;

        for i = 1:m
            res = 0;
            for j = 1:feature_num
                res = res + theta(j)*X(i,j);
            end;
            res = (res - y(i))*X(i,itheta);
            sigma += res;
        end;

        next_theta(itheta) = theta(itheta) - alpha/m*sigma;
    
    end;

    theta = next_theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
