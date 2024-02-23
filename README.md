# RCML
This repository contains the code of our AAAI2024 paper Reliable Conflictive Multi-view Learning.

# Background
Multi-View data obtained from multiple sources or different feature subsets. Information gathered from different sources is likely to be conflicting.
This conflictive information in different views makes most multi-view learning methods inevitably degenerate or even fail.
Conflictive instances contain noise and unalignment views.
![image](https://github.com/jiajunsi/RCML/assets/92369008/bbb3992a-298e-44d1-a301-b8a2f2da00e9)

# Motivations
Previous methods aim to eliminate conflictive instances, while realworld applications usually require making decisions for them. 
Considering the decision of a conflictive instance might be unreliable, we need the model can answer “should the decision be reliable?”.
![image](https://github.com/jiajunsi/RCML/assets/92369008/a47e3ca0-cb06-4be5-9c4f-28dd24c9fe2e)

# Model
![image](https://github.com/jiajunsi/RCML/assets/92369008/84818820-c733-4f0e-88ba-1634a3172cc9)

# Experiments
![image](https://github.com/jiajunsi/RCML/assets/92369008/da7a597b-a985-43e6-9500-c53d116248d4)
![image](https://github.com/jiajunsi/RCML/assets/92369008/14f24165-8a71-48d0-8a09-7c36513f10d5)

# Future Work
1、 Conflicting multi-view data is very common in the real world, but there are currently no datasets related to it, and we are working to collect data sets and share them with everyone. 
2、The proposed average aggregation schema may not be the optimal solution. For example, the specific order in which fusion takes place can influence the final result.
3、 For different data sets, the results vary greatly, and we have a hypothesis that this may be related to the differences between sample categories and the number of sample views
