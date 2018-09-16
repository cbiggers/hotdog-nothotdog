function [acc prec rec f] = predictionMetrics(y, p)
  
tp = sum((y==1)(p==1)); ## true positives
tn = sum((y==0)(p==0)); ## true negatives
fp = sum((y==0)(p==1)); ## false positives
fn  = sum((y==1)(p==0)); ## false negatives


## Octave may have warnings for /0, let's avoid
acc = (tp+tn) / length(y);

if (tp+fp) == 0,
  prec = 0;
else prec = tp / (tp + fp);
end

if (tp + fn) ==0,
  rec = 0;
else rec = tp / (tp+fn);
end

if (prec+rec) == 0,
  f = 0;
else f = 2 * ((prec*rec)/(prec+rec));
end  
  
end