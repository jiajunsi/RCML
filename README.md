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

# Related work
There are many interesting works inspired by this paper and the following are related follow-up works: <br>
\[1\] [Zongbo Han, Changqing Zhang, Huazhu Fu and Joey Tianyi Zhou. Trusted multi-view classification. In International Conference on Learning Representations, 2021.](https://arxiv.org/abs/2102.02051) <br>
\[2\] [Murat Sensoy, Lance Kaplan and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Neural Information Processing Systems, 31, 2018.](https://dl.acm.org/doi/abs/10.5555/3327144.3327239) <br>
\[3\] [Gawlikowski J, Tassi C R N, Ali M, et al. A survey of uncertainty in deep neural networks. In Artificial Intelligence Review, 56(Suppl 1): 1513-1589, 2023.](https://link.springer.com/article/10.1007/s10462-023-10562-9) <br>
\[4\] [Danruo Deng, Guangyong Chen, Yang Yu, Furui Liu, and Pheng-Ann Heng. Uncertainty estimation by fisher information-based evidential deep learning. In International Conference on Machine Learning, Vol. 202. JMLR.org, Article 300, 7596–7616, 2023.](https://dl.acm.org/doi/10.5555/3618408.3618708) <br>
\[5\] [Jøsang, Audun. Subjective logic. Cham: Springer, 2016.](https://link.springer.com/book/10.1007/978-3-319-42337-1) <br>
\[6\] [J. Liu, X. Liu, Y. Yang, Q. Liao and Y. Xia. Contrastive Multi-View Kernel Learning. In IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 9552-9566, 2023.](https://ieeexplore.ieee.org/abstract/document/10061269) <br>
\[7\] [Zhenyu Huang, Peng Hu, Joey Tianyi Zhou, Jiancheng Lv and Xi Peng. Partially view-aligned clustering. In Neural Information Processing Systems, 33: 2892-2902, 2020.](https://proceedings.neurips.cc/paper/2020/hash/1e591403ff232de0f0f139ac51d99295-Abstract.html) <br>
\[8\] [Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. In Neural Information Processing Systems, 31, 2018.](https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html) <br>
\[9\] [W. Dong and S. Sun. Multi-View Deep Gaussian Processes for Supervised Learning. In IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 15137-15153, 2023.](https://ieeexplore.ieee.org/abstract/document/10255358) <br>
\[10\] [J. Wen et al. A Survey on Incomplete Multiview Clustering. In IEEE Transactions on Systems, Man, and Cybernetics: Systems, pp. 1136-1149, 2023.](https://ieeexplore.ieee.org/abstract/document/9845473) <br>
