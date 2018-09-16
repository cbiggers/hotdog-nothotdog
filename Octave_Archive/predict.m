function [p cutoff] = predict(theta_params,X,y,n,h,num_labels,cutoff)

m = size(X)(1);
p = zeros(m,1);

## unroll theta parameters
theta1 = reshape(theta_params(1:((n+1)*(h(1))),:), n+1, h(1)); ## n+1 by h
theta2 = reshape(theta_params((numel(theta1)+1):((numel(theta1))+((h(1)+1)*h(2))),:),h(1)+1,h(2));
theta3 = reshape(theta_params((numel(theta2)+numel(theta1)+1):end,:),h(2)+1,num_labels);

h1 = sigmoid(X * theta1); ## [m by n+1]*[n+1 by h(1)] >> [m by h(1)]
h2 = sigmoid([ones(m,1) h1] * theta2);   ## [m by h(1)+1]*[h(1)+1 by h(2)] >> [m by h(2)]
h3 = sigmoid([ones(m,1) h2] * theta3); ## [m by h(2)+1] * [h(2)+1 by num_labels]

best_f = 0;
if cutoff==0,
  for i = .01:.005:.60,
    p = (h3>=i);
    [acc prec rec f] = predictionMetrics(y,p);
    if f > best_f, 
      best_f = f;
      cutoff = i;
    end 
  end 
end

p = (h3>=cutoff);

end