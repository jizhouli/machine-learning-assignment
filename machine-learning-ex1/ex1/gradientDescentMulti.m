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

	% sigma = 0;
	% for i = 1:m
	% 	sigma = sigma + theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3) - y(i);
	% end;
	% theta_0 = theta(1) - alpha / m * sigma;

	% sigma = 0;
	% for i = 1:m
	% 	sigma = sigma + (theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3) - y(i)) * X(i,2);
	% end;
	% theta_1 = theta(2) - alpha / m * sigma;

	% sigma = 0;
	% for i = 1:m
	% 	sigma = sigma + (theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3) - y(i)) * X(i,3);
	% end;
	% theta_2 = theta(3) - alpha / m * sigma;

	% theta = [theta_0; theta_1; theta_2];

	feature_num = size(X, 2);
	for i = 1:m
		res = 0;
		% caculate H_theta_function_result
		for j = 1:feature_num
			res = res + theta(j)*X(i,j);
		end;
		res = res - y(i);
		% update theta
		for j = 1:feature_num
			theta(j) = theta(j) -alpha / m * (res * X(i,j));
		end;
	end;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
