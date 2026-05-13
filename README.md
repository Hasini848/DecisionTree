DECISION TREE
To check purity of the splits we have two types:
 Entropy and ginni index
The feature we select for splitting is Information Gain
Entropy: h(s)=-p+log2p+ - p-log2p-
Ginni Index: G.I=1-summation(p)^2
For entropy we get values between 0 and 1 
if there is small dataset we will use entropy
if there is large dataset we will use ginni index
Pruning is of 2 types:
a. Post-pruning: we will construct the entire decision tree then will prun using hyper parameter maxdepth.
b. Pre-pruning: by uing hyper parameter we can achevie this.