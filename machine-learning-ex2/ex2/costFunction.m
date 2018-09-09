function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% calculate omega
omega = 0.0;
for i = 1:m
    % calculate polynomial
    poly = X(i,1)*theta(1) + X(i,2)*theta(2) + X(i,3)*theta(3);
    % calculate sigmoid of polynomial
    sigm = sigmoid(poly);
    omega = omega + (y(i)*log(sigm) + (1-y(i))*log(1-sigm));
end;
% calculate cost
J = omega*(-1)/m;

for j = 1:length(theta)
    % calculate omega
    omega = 0.0;
    for i = 1:m
        % calculate polynomial
        poly = X(i,1)*theta(1) + X(i,2)*theta(2) + X(i,3)*theta(3);
        % calculate sigmoid of polynomial
        sigm = sigmoid(poly);
        omega = omega + (sigm-y(i))*X(i,j);
    end;
    grad(j) = omega/m;
end;


% =============================================================

end
