function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

sigma = 0.0;
for i = 1:m
    poly = 0.0;
    for k = 1:length(theta)
        poly = poly + X(i,k)*theta(k);
    end;
    sigm = sigmoid(poly);
    sigma = sigma + (y(i)*log(sigm) + (1-y(i))*log(1-sigm));
end;
J = sigma * (-1)/m;

reg_weight = 0.0;
sigma = 0.0;
for j = 2:length(theta)
    sigma = sigma + theta(j)^2;
end;
reg_weight = lambda/(2*m)*sigma;

J = J + reg_weight;




% calculate gradient of theta0, and theta1-n
for j = 1:length(theta)
    sigma = 0.0;
    for i = 1:m
        poly = 0.0;
        for k = 1:length(theta)
            poly = poly + X(i,k)*theta(k);
        end;
        sigm = sigmoid(poly);
        sigma = sigma + (sigm-y(i))*X(i,j);
    end;
    grad(j) = sigma/m;

    if j>=2
        grad(j) = grad(j) + lambda/m*theta(j);
    end;
end;

% =============================================================

end
