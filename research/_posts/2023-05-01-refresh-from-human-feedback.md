---
layout: post
title: "Refresh From Human Feedback"
date: 2023-04-01
category: research
author: "Tong Guo"
description: "Refresh From Human Feedback"
---
# Refresh From Human Feedback

## Abstract

In industry deep learning application, our main goal is to get a high quality training dataset. Reinforcement Learning from Human Feedback (RLHF) trains a reward model to guide policy model. But the problem is that RLHF only use the reward model's dataset. Then the policy model' dataset is lose. To solve this problem we propose a more efficient method:  First, the trained policy model predicts for a new test data and get its model result. Then, if the model result is wrong by human feedback, then we use the model' output embedding to find the one most similar training data and remove it from training dataset. (Or we can find and remove all the most similar training data to the wrong model result of test data by BM25 search method.) The reason is that if the model result of the test data is wrong, then the one most similar training data of the new test data must be wrong. If the model result is right by human feedback, then we merge the new test data with model's label to training dataset. By doing this, we simply the human feedback work to 0-or-1 to label, which improve the labeling efficiency and accuracy.

## Introduction

## Method

![fig1](/assets/png/refresh/fig1.png)


Step-5 should be in front of Step-6, because the badcase removing Step-6 is more essential than merging Step-5. We should first merge the reviewed data to the training dataset, and then we search all the most similar data of badcase data in the merged training dataset.

## Experiments

We evaluate our method on text generation task. 

## Discussion
For text generation task, the correction Step-7 can be a choice question for labeling people. The labeling people chooses the best result from all the candidates: The origin result of training data and the model results.

In industry application, we evaluate the model predictions of online dataset, for every period of time. Using the method of this paper, we can collect the evaluation results to improve the model. So the evaluation results are not wasted.
 
## References

```
[1] Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. Advances in Neural Information Processing Systems, 2022, 35: 27730-27744.
```
