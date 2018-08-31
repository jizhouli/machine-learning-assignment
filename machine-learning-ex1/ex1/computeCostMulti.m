function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

feature_num = length(theta); %size(X, 2);
for i = 1:m
	poly_res = 0; % polynomial function of each line/raw
	for j = 1:feature_num
		% fprintf('%d, %d: X(i,j)=%f, theta(j)=%f, poly_res=%f \n', i,j,X(i,j),theta(j),poly_res);
		poly_res = poly_res + X(i,j)*theta(j);
	end;
	J = J + (poly_res - y(i))^2;
end;

J = J /(2*m);
% =========================================================================

end
