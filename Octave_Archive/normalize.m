function [x_norm] = normalize(x)
## standard mean normalization and feature scaling done for 
## most learning algorithms

mu = mean(x);
x = x - mu;
sigma = std(x);
x_norm = x./sigma;


end