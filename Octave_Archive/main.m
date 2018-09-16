## 
##  Main routine training hotdog/nothotdog algorithm
##

## Initialize
clear ; close all; clc

load('data_train.mat');
load('data_valid.mat');
load('data_test.mat');

### Parameters to input ####
h = [300 50];
num_labels = 1;
lambda = 1;
jump = ceil(size(data_train)(1)/10); ## jump size for m when building the learning curve

#############################

## Set up initial variables ##
m_train = size(data_train)(1);
m_valid = size(data_valid)(1);
m_test = size(data_test)(1);

x_train = normalize(data_train(:,1:end-1));
x_valid = normalize(data_valid(:,1:end-1));
x_test = normalize(data_test(:,1:end-1));

y_train = data_train(:,size(data_train)(2));
y_valid = data_valid(:,size(data_valid)(2));
y_test = data_test(:,size(data_test)(2));

## n is initial number of features, not including intercept
n = size(x_train)(2); 

x_train = [ones(m_train,1) x_train];  ## add in intercept feature
x_valid = [ones(m_valid,1) x_valid];  ## add in intercept feature
x_test = [ones(m_test,1) x_test];  ## add in intercept feature

bias = (m_train-sum(y_train))/sum(y_train);
## initialize random thetas
## theta1 --> maps L1 to L2. n+1 units to h units, so n+1 by h. 
## theta2 --> Maps L2 to K. h+1 units map to 1. 

#theta1 = stdnormal_rnd(n+1,h);
#theta2 = stdnormal_rnd(h+1,num_labels);
#theta_params = [theta1(:) ; theta2(:)];

## Example cost function use
##[J grad_list] = costFxn(theta_params, n, h, num_labels, x_train, y_train, lambda); 

##### Training the neural network ######

#{
## first, set options for fmincg(advanced optimization). optionally iterate over this. 
options = optimset('MaxIter', 50);
## can optionally iterate over lambda
## Create "short hand" for the cost function to be minimized, used by fmincg
costFunction = @(p) costFxn(p, n, h, num_labels, x_train, y_train, lambda);
[theta_params, cost] = fmincg(costFunction, theta_params, options);
## cost is a list of each iteration and its cost
#}

#### Initial train of regression to get theta parameters ####
[theta_params] = trainReg(n,h,num_labels,x_train,y_train,lambda,bias);

####### Metrics / predictions on Train/test/valid sets #######



## use validation set to set prediction cutoff (>.1 is hotdog? >.2 is hotdog?)
[p_train cutoff] = predict(theta_params, x_train, y_train, n, h, num_labels,0);
p_valid = predict(theta_params, x_valid, y_valid,n, h, num_labels,cutoff);
p_test = predict(theta_params, x_test, y_test,n, h, num_labels,cutoff);


#{
Accuracy: How many are predicted correctly?
Precision: Of all predicted positives, how many are true?
Recall: Of all actual positives, how many did we predict?
#}

#[y_test p_test];
fprintf('Training metrics...\n')
[acc prec rec f] = predictionMetrics(y_train,p_train) 
fprintf('Validation metrics...\n')
[acc prec rec f] = predictionMetrics(y_valid,p_valid)
fprintf('Testing metrics...\n')
[acc prec rec f] = predictionMetrics(y_test,p_test)


###Learning Curve Implementation###
#{
[train_costs valid_costs test_costs] = learningCosts(x_train,y_train,x_valid,y_valid, ...
                                                      x_test,y_test,n,h,lambda,jump,bias);
plot(1:jump:m_train,train_costs,1:jump:m_train,valid_costs,1:jump:m_train,test_costs);
title('Learning Curve for Hotdog / NotHotdog Network')
legend('Train','Validation')
xlabel('# of training examples')
ylabel('Cost')
#}
