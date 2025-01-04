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


### 2. Method

![fig1](/assets/png/self-refine-use-llm/fig1.png)

### 3. Experiment

### 4. Discussion

### 5. Relate Work

### 6. Conclusion

### Reference
