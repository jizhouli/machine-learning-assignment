function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i = 1:m
    polynomial_result = 0; % theta(1);
    for j = 1:length(theta)
        polynomial_result = polynomial_result + X(i,j)*theta(j);
        % fprintf('%d, %d: %f, %f \n', i, j, polynomial_result, y(i));
    end;
    J = J + (polynomial_result - y(i))^2;
	% fprintf('%d: pr=%f, y(i)=%f, J=%f\n', i, polynomial_result, y(i), J);
    % fprintf('J is %f \n', J);
end;

J = J / (2*m);
% fprintf('J = %f \n', J);


% =========================================================================

end
