---
layout: post
title: "Simple Self-Eval-Drop for Leveraging LLMs"
date: 2024-09-01
category: research
author: "Tong Guo"
description: "Simple Self-Eval-Drop for Leveraging LLMs"
---
# Simple Self-Eval-Drop for Leveraging LLMs

### Abstract

Leveraging large language models (LLMs) to performs direct inference or build training datasets has becoming an important method for natural language processing (NLP) applications. However the quality of the training datasets or inference results of LLMs exist some noise data. In this paper, we propose our self-eval-drop method to clean the datasets by LLMs API. We first sort the to-clean dataset and then sample a seed dataset from the to-clean dataset. Then we manually evaluate the seed dataset and find badcases in the seed dataset. Then we loop the badcases to find the (batch) similar data in the to-clean dataset for each badcase. The we remove the found similar data of badcases from the to-clean dataset. The manual evaluation results show that our simple self-eval-drop method can improve the accuracy of the to-clean dataset to more than 97% by only sampling and evaluating about 1% of the to-clean dataset as the seed dataset.

### Introduction

In recent years, the development of large language models (LLMs) [1] has brought breakthroughs on NLP applications. However the quality of the training datasets or inference results of LLMs exist some noise data for the specific deep learning (DL) [2] applications. In order to solve the noise data problem, we propose our self-eval-drop method to clean the datasets by LLMs API.

(some examples to leverage LLMs API)

### Method

### Manual Evaluation

### Discussion

### Relate Work

### Conclusion

### Reference
```
[1] Achiam J, Adler S, Agarwal S, et al. Gpt-4 technical report[J]. arXiv preprint arXiv:2303.08774, 2023.

[2] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25.
```
