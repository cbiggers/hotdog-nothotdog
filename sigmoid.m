function g = sigmoid(z)
## cost function is sigmoid for neural network
g = 1.0 ./ (1.0 + exp(-z));
## replace with reLU
#g = max(0,z);

end