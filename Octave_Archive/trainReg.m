function [theta_params cost] = trainReg(n,h,num_labels,x,y,lambda,bias)
  
#{  
## initialize random thetas
## theta1 --> maps L1 to L2. n+1 units to h units, so n+1 by h. 
## theta2 --> Maps L2 to K. h+1 units map to 1. 
#theta1 = stdnormal_rnd(n+1,h);
#theta2 = stdnormal_rnd(h+1,num_labels);
#}

#### Currently -- hard-coded for 4-layer neural network ########
## 1(Input) -> 2(1st hidden) -> 3(2nd hidden) -> 4 (y)
theta1 = zeros(n+1,h(1)); ## each weighted feature mapped to each node in first hidden layer
theta2 = zeros(h(1)+1,h(2)); ## each node of 1st hidden layer mapped to node of 2nd hidden layer
theta3 = zeros(h(2)+1,num_labels); 
theta_params = [theta1(:) ; theta2(:) ; theta3(:)];

## standard call to cost fxn
##[J grad_list] = costFxn(theta_params, n, h, num_labels, x, y, lambda); 

##### Training the neural network ######
## first, set options for fmincg(advanced optimization). optionally iterate over this. 
options = optimset('MaxIter', 50);

## Create "short hand" for the cost function to be minimized, used by fmincg
costFunction = @(p) costFxn(p, n, h, num_labels, x, y, lambda,bias);

[theta_params, cost] = fmincg(costFunction, theta_params, options);
end