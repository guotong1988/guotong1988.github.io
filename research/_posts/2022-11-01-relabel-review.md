---
layout: post
title: "Re-Label Method: A Review"
date: 2022-11-01
category: research
comments: true
author: "Tong Guo"
description: "Re-Label For Industry 97% Accuracy: A Review"
---


# Re-Label Method: A Review

## Abstract
Recently, the fast development of deep learning has brought computer science to a new era. 
In this survey, We first summarize how to achieve 97% accuracy in any human-labeled industry deep learning application in both dev dataset and human evaluation by re-label / label-again, 
based on the principle that deep learning is rule-injected model with strong generalization ability. 
Then we extend the re-label method to more possible idea.
In the end, we imagine re-label method to further impossible idea.

#### Keywords

Deep Learning, Pattern Recognition, Fuzzy Matching, Similar Search, Data Centric, Human Labeling

## 1. Introduction

Deep learning is the deep neural network that has the most strong generalization ability now. 
And the generalization ability is based on the labeled training data. The training data is injected by human knowledge/rule.
Data centric methods without changing the ground-truth-label do not have enough effect to the model performance.
Re-Label Methods \cite{ref1} \cite{ref2} \cite{ref3} correct the noisy data and re-direct the wrong knowledge data,
and achieve 97% accuracy in any human-labeled deep learning application. 




## 2. Further Idea

### 2.1 Combine Re-label Method and Reinforcement Learning


#### 2.1.1 With Human Correction, Reward Is Enough 

Human learn from base concept and get 'reward' from the physical world, 
in the same time human correct and refresh the knowledge from new 'reward'.

Keeping the knowledge of robot/agent to be always right and fresh is important. 
The self-correction for robot/agent is very hard.
The correction can be done by human. 
Human change the base reward rule for robot/agent to refresh their knowledge. 



## References
```
\bibitem{ref1}
Guo T. Learning From Human Correction For Data-Centric Deep Learning[J]. arXiv preprint arXiv:2102.00225, 2021.

\bibitem{ref2}
Guo T. Re-Label Is All You Need[J].

\bibitem{ref3}
Guo, Tong (2022): Re-Label By Data Pattern Is All You Need For Knowledge Driven Deep Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.20485917.v3 
```
