function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[rows, cols] = size(z);
for r = 1:rows
    for c = 1:cols
        val = z(r,c);
        % do sigmoid calculation on scalar value val
        g(r,c) = 1 / (1 + e^(-val));
    end;
end;

% =============================================================

end
