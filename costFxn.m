function [J grad_list] = costFxn(theta_params,...
                            n, h, num_labels,...
                            x, y, lambda,bias)
                            
#{        
## cost function for neural network
## returns cost J and gradient for a single input of theta weights. 
## (N): need to take theta parameters in one unrolled list due to how  
## the advanced optimization algorithm fmincg works
#}


##Initial setup
m = size(x)(1);
# x = x(ones(m,1) x);   #bias term already included 
J = 0;
grad_list = 0;

## re-form theta1 and theta2 from theta_params
theta1 = reshape(theta_params(1:((n+1)*(h(1))),:), n+1, h(1));
theta2 = reshape(theta_params((numel(theta1)+1):((numel(theta1))+((h(1)+1)*h(2))),:),h(1)+1,h(2));
theta3 = reshape(theta_params((numel(theta2)+numel(theta1)+1):end,:), h(2)+1, num_labels);
theta1_grad = zeros(size(theta1)); ## n+1 by h(1)
theta2_grad = zeros(size(theta2)); ## h(1)+1 by h(2)
theta3_grad = zeros(size(theta3)); ## h(2)+1 by num_labels

######################################
## Forward-propogation ###############
#{
Notation: 
  a -> nodal values (sigmoid function applied to weighted values)
  z -> weighted values before application of sigmoid function
#}

a1 = x;    ## [m by n+1]
z2 = theta1' * a1'; ## [n+1 by h(1)]'  * [m x n+1]' >> h(1) x m
a2 = sigmoid(z2); ## [h(1) by m]
a2 = [ones(1,m) ; a2]; ## [h(1)+1 by m], creating bias term for a2
z3 = theta2' * a2;   ## [h(1)+1 by h(2)]' * [h(1)+1 by m] >> [h(2) by m]
a3 = sigmoid(z3); ## [h(2) by m]
a3 = [ones(1,m) ; a3]; ##[h(2)+1 by m]
z4 = theta3' * a3; ##[h(2)+1 by labels]' * [h(2)+1 by m] >> [labels by m]
hx = sigmoid(z4)'; ## [labels by m]' 

##[m by num_labels]

#### Structure y in same structure as h(x)######
## unnecessary here because only one label, 0/1 already. 

############################
## can now compute cost ####
## note that it's more complex with summing over multiple K, need loop in that case

# size(hx) = m 1  
# size(y) = m 1
## add bias term here to give more weight to the positive observations
## since data set is skewed (way more not hot dogs than hot dogs)
J = sum(((y.*log(hx))) + ((1-y).*log(1-hx)));
J = (-1/m)*J;
## regularize
reg_weight = sum(sum(theta1.^2)) + sum(sum(theta2.^2)) + sum(sum(theta3.^2));
reg_weight = reg_weight - sum(theta1(1,:).^2) - sum(theta2(1,:).^2) - sum(theta3(1,:).^2); ## don't regularize bias terms
J = J + (lambda/(2*m))*reg_weight;
####Backpropagate to find gradients#######
##########################################

#{
Calculate weight of errors introduced at each node for each m. 
Use these to determine the next gradient step. 
#}

## Step #1: remove bias terms since they don't introduce error
theta3 = theta3(2:end,:); ## [h(2)+1-1 by 1];
theta2 = theta2(2:end,:); ## [h(1)+1-1 by h(2)];

s4 = hx - y;  ## error term for each m for each label
s3 = (theta3*s4') .* sigmoidGradient(z3); ## [h(2) by 1] * [m by 1]' .* [h(2) by m]
s2 = (theta2*s3) .* sigmoidGradient(z2); ## [h(1) by h(2)] * [h(2) by m] >> [h(1) by m] .* [h(1) by m]
theta3_grad = (a3 * s4); ##[h(2)+1 by m] * [m by 1] >> [h(2)+1 by 1]
theta2_grad = (s3 * a2'); ##[h(2) by m] * [h(1)+1 by m]' >> [h(2) by h(1)+1]
theta1_grad = (s2 * a1); ## [h(1) by m] * [m by n+1] >> [h(1) by n+1]
#{
#######################
s3 = hx - y;  ## s3 = error term for each label for each example m, is m x 1
## s2 = error for each example m at each unit of the hidden layer h
## so s2 should be 100x22  
theta2 = theta2(2:end,:);      
s2 = (theta2*s3') .* sigmoidGradient(z2); #h by m .* h x m
theta2_grad = (s3' * a2)'; ## (m x 1)' * (m x h+1) >> h+1 by 1
theta1_grad = (s2 * a1)'; ## (h x m) * (m * n+1) >> n+1 by h
#}


## regularize gradients
## we do not regularize the gradient for bias terms
## so replace these thetas along with '0' for regularization calculations
grad_list  = [theta1_grad(:) ; theta2_grad(:) ; theta3_grad(:)];
#################
reg_theta1 = [zeros(h(1),1) theta1_grad(:,2:end)]; ## [h(1) by n+1]
reg_theta2 = [zeros(h(2),1) theta2_grad(:,2:end)]; ## [h(2) by h(1)+1
reg_theta3 = [zeros(1,1) ; theta3_grad(2:end,:)]; ## [h(2)+1 by 1]
reg_theta  = [reg_theta1(:) ; reg_theta2(:) ; reg_theta3(:)];
#size(grad_list)
#size(theta1_grad)
#size(reg_theta1)
#size(reg_theta)
grad_list = (grad_list .* (1/m)) + ((lambda/m).*reg_theta)  ;

end