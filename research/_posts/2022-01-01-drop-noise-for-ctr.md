---
layout: post
title: "Drop Noise For CTR Prediction"
date: 2022-01-01
category: research
comments: true
author: "Tong Guo"
description: "Drop Noise For CTR Prediction"
---


# Drop Noise For CTR Prediction

### abstract

Click-through rate (CTR) prediction is task to estimate the possibility of user clicking on a recommended item. The ground truth labels of this task are the click history of users. As there are many reasons why noise data or low quality data may be generated, the training data has a certain amount of noise data or low quality data. In this work, We propose a simple but effective method to find the noise data. Our method could improve the offline AUC from 0.60 to 0.75 on our real-world dataset.

#### keywords

CTR, Recommendation System, Deep Learning


### Introduction

The prediction of click-through rate is very important in recommendation systems, and its task is to estimate the possibility of user clicking on a recommended item. In recommendation systems the goal is to maximize the CTR, so the item list for a user should be ranked by estimated CTR.

The ground truth labels of this task are the click history of users. There are many reasons that may generate noise or low quality training data. We can not clearly define what are the noise data. But our method can find the noise data that harm the performance of our model.

Previous works \cite{ref1} \cite{ref2} \cite{ref3} have shown the effectiveness of our method on human-labeled dataset. In CTR task, the training data is generated by user behavior. So if the amount of user behavior training data is large enough, we can remove the noise data and do not need to correct them. 


### Related Work

There are many works focus on the model-centric perspective of CTR task. Factorization Machines (FM) \cite{ref4} , DeepFM \cite{ref5}, Wide \& Deep model \cite{ref6} are all works that solve the model-centric perspective of this task.

Previous works \cite{ref1} \cite{ref2} \cite{ref3} focus on the data-centric perspective on human-labeled dataset and do not apply the idea to user-generated dataset like CTR task.
 

### Method

![](/assets/png/drop-ctr/fig1.png)

In this section, we describe our method in detail. Our methods is shown in Fig 1. It includes 5 steps:

Step 1, in order to solve our industry CTR problem. We get the user-generated dataset-A and prepare features like age, gender, location. 

Step 2, we train CTR deep model on dataset-A. We named the result model of this step Model-A. Note that Model-A should not overfit the training dataset.

Step 3, we use Model-A to predict for all the dataset-A. Then we find all the data whose user-generated label and predicted-label is different. We consider they are the noise data. There are many ways to define the difference of two label: the equality of the two label, or the distance of the two score.

Step 4, we remove the noise data, as we have enough user-generated data. Then we get dataset-B.

Step 5, we train upon the dataset-B and get Model-B.


### Experiments

In this section we describe detail of experiment parameters and show the experiment result.

#### Experiment Result

In this section, we evaluate our methods on our real-world dataset. Our dataset-A contains 2,000,000,000 user-item click-or-not data and each data has 100 features. Table 1 shows the performance comparison on the dataset. The model is DeepFM. As our method is data-centric approach, we do not focus on which model we use.

### Analysis

In this section, we illustrate why drop the noise data work.

Take text classification as example. If there are 2-class to classify. The sample training data is like: 


---


'aac' -- class-A

'aad' -- class-A

'aae' -- class-B -- wrong-label

'bbc' -- class-B 

'bbd' -- class-B 

'bbe' -- class-B 


---


We define that the text started with 'aa' to label class-A and started with 'bb' to label class-B.

Then the trained model will inference 'aae' to class-A, but will inference 'aaee' to class-B, according to the training data distribution.

Then we drop the wrong-label training data and get:


---


'aac' -- class-A

'aad' -- class-A

'bbc' -- class-B 

'bbd' -- class-B 

'bbe' -- class-B 


---


Then the trained model will inference 'aae' to class-A, and also will inference 'aaee' to class-A, according to the training data distribution.

The reason for this example is: A little wrong-label data can lead to wrong inference for some kind of new data, but the wrong-label data can be found by self-predict-and-compare method.


### Conclusion

Based on the good performance of previous works  \cite{ref1} \cite{ref2} \cite{ref3} that have been verified on human-labeled dataset. We further apply the find-noise idea to user-generated dataset and CTR task. The experiment result shows our idea could improve the AUC a lot. As recommendation system predicts the rating or the preference a user might give to an item. Or it is an algorithm that suggests relevant items to users. The most important thing is to find a way to fit the large amount of data. In other words, fitting the dataset means we find the preference of users. Also, noise data is not the low-frequency user data. We will verify our idea on online performance in the future.  


![](/assets/png/drop-ctr/table1.png)


### Reference

\bibitem{ref1}

Guo T. Learning From Human Correction For Data-Centric Deep Learning[J]. arXiv preprint arXiv:2102.00225, 2021.

\bibitem{ref2}

Guo, Tong (2021): Self-Refine Learning For Data-Centric Text Classification. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.16610629.v3 

\bibitem{ref3}

Guo, Tong (2021): Achieving 90% In Data-Centric Industry Deep Learning Task. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.17128475.v2 

\bibitem{ref4}

[Rendle, 2010] Steffen Rendle. Factorization machines. In ICDM, 2010.

\bibitem{ref5}

Guo H, Tang R, Ye Y, et al. DeepFM: a factorization-machine based neural network for CTR prediction[J]. arXiv preprint arXiv:1703.04247, 2017.

\bibitem{ref6}

Cheng H T, Koc L, Harmsen J, et al. Wide \& deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.

