---
layout: post
title: "Self-Refine Learning For Cleaning LLMs Data"
date: 2024-12-31
category: research
author: "Tong Guo"
description: "Self-Refine Learning For Cleaning LLMs Data"
---
# Self-Refine Learning For Cleaning LLMs Data

### Abstract

In industry NLP application, our dataset by prompting large language models (LLMs) has a certain number of noise data. 
We present a simple method to find the noise data and remove them. 
We retain the data that contains certain common tokens between the LLMs data and the prediction results of a generative model trained on the LLMs data.
We remove the data that does not contain certain common tokens between the LLMs data and the prediction results of a generative model trained on the LLMs data.
We adopt T5-Base as our generative model.
The experiment result shows our method is highly effective and **does not require any manual annotation**.
For industry deep learning application, our method improves the NLP tasks accuracy from 88% to 98% under human evaluation, meanwhile the LLMs data source is sufficiently abundant.

### 1. Introduction
With the development of LLMs, prompting LLMs to obtain a domain-specific training dataset and then training a smaller model to achieve sufficiently fast model inference has become a very useful approach for performing NLP tasks. However, the accuracy of LLMs' original datasets within a specific domain generally only reaches 88%-90%. Additionally, using manual annotation methods to correct or clean the datasets requires a substantial amount of human resources and time. Therefore, this paper proposes a method for automatic dataset cleaning. Experiments show that it is highly effective, requires no manual annotation, and can be extended to more AI tasks based on LLMs datasets, such as computer vision.

### 2. Method

![fig1](/assets/png/self-refine-use-llm/fig1.png)

![alg1](/assets/png/self-refine-use-llm/alg1.png)

### 3. Experiment

![table1](/assets/png/self-refine-use-llm/table1.png)

### 4. Discussion

In actual observations, we found that almost all noisy data with significant issues were filtered out. The significant issues here refer to the presence of unexpected special tokens in the data.

This automated cleaning method is a highly compatible combination with the method of obtaining data through prompting LLMs. The amount of training data obtained via prompting LLMs is sufficient, and we can set the necessary control thresholds to filter the data.

### 5. Relate Work

### 6. Conclusion

### Reference
