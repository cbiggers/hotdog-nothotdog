function [train_errors val_errors test_errors] = learningCosts(x_train,y_train,x_valid,y_valid,...
                                                  x_test,y_test,n, h, lambda,jump,bias)
  
m = size(x_train)(1);
#train_errors = zeros(ceil(m/jump),1);
#val_errors = zeros(ceil(m/jump),1);
#test_errors = zeros(ceil(m/jump),1);
num_labels = size(y_train)(2);
cnt = 0;

for i = 1:jump:m,  ## in steps
  cnt = cnt+1;
  x = x_train(1:i,:);   ##[i by n+1]
  y = y_train(1:i); ## [i by 1]
  ## learn thetas for this dataset size. 
  [theta_params] = trainReg(n,h,num_labels,x,y,lambda,bias);
  ## compute costs, given these learned thetas, at the training subset size (i:m) 
  ## for the entire training set and the entire testing set. 
  ## Note that lambda should be 0 as we've already trained the theta parameters with a L!
  [J_train g] = costFxn(theta_params,n,h,num_labels,x,y,0,bias);
  [J_val g] = costFxn(theta_params,n,h,num_labels,x_valid,y_valid,0,bias);
  [J_test g] = costFxn(theta_params,n,h,num_labels,x_test,y_test,0,bias);
  #fprintf('Train Cost: %d \n',J_train)
  #fprintf('Valid Cost: %d \n',J_val)
  train_errors(cnt) = J_train;
  val_errors(cnt) = J_val;
  test_errors(cnt) = J_test;
end

  
end