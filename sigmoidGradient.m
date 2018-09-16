function g = sigmoidGradient(z)
  
#### returns the gradient of the sigmoid function #####

g = zeros(size(z));

s = 1 ./ (1 + exp(-z));
g = s .* (1-s);
end