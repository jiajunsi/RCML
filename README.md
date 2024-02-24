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
1、 Conflicting multi-view data is very common in the real world, but there are currently no datasets related to it, and we are working to collect data sets and share them with everyone. <br> 
2、The proposed average aggregation schema may not be the optimal solution. For example, the specific order in which fusion takes place can influence the final result. <br>
3、 For different data sets, the results vary greatly, and we have a hypothesis that this may be related to the differences between sample categories and the number of sample views. <br>

# 相关工作
1、Han Z, Zhang C, Fu H, et al. Trusted Multi-View Classification[C]//International Conference on Learning Representations. 2020. <br>
2、Sensoy M, Kaplan L, Kandemir M. Evidential deep learning to quantify classification uncertainty[J]. Advances in neural information processing systems, 2018, 31. <br>
3、Gawlikowski J, Tassi C R N, Ali M, et al. A survey of uncertainty in deep neural networks[J]. Artificial Intelligence Review, 2023, 56(Suppl 1): 1513-1589. <br>
4、Danruo D, Chen G, Yang Y U, et al. Uncertainty Estimation by Fisher Information-based Evidential Deep Learning[J]. 2023. <br>
5、Jøsang A. Subjective logic[M]. Cham: Springer, 2016. <br>
6、Liu J, Liu X, Yang Y, et al. Contrastive Multi-View Kernel Learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. <br>
7、Huang Z, Hu P, Zhou J T, et al. Partially view-aligned clustering[J]. Advances in Neural Information Processing Systems, 2020, 33: 2892-2902. <br>
8、Malinin A, Gales M. Predictive uncertainty estimation via prior networks[J]. Advances in neural information processing systems, 2018, 31. <br>
9、Dong W, Sun S. Multi-View Deep Gaussian Processes for Supervised Learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. <br>
10、Wen J, Zhang Z, Fei L, et al. A survey on incomplete multiview clustering[J]. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 53(2): 1136-1149. <br>
