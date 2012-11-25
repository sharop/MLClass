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

h = sigmoid(X*theta);
subterm1 = -y.*log(h);
subterm2 = (1-y).*log(1-h);
sumatoria = sum(subterm1-subterm2);
term1 = sumatoria/m;
taux = theta;
taux(1)=0; % No se regulariza el termino 1 de theta.
term2 = (sum(taux.^2)/(2*m))*lambda;
J=term1+term2;
grad = ((1/m).*(h.-y)'*X) + ((lambda/m).*taux)';


% =============================================================

end