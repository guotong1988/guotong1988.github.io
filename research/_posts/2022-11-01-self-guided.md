---
layout: post
title: "Self-Guided Human Labeling For Industry Deep Learning"
date: 2022-11-01
category: research
comments: true
author: "Tong Guo"
description: "Self-Guided Human Labeling For Industry Deep Learning"
---


# Self-Guided Human Labeling For Industry Deep Learning

### Abstract

In human-labeled industry deep learning application, the human-labeled data quality determine the model performance. 
To gurantee the data quality, we propose our self-guided or self-reference method. 
We first manually label a high quality small seed dataset.
Then we train model upon the seed dataset.
Then we predict labels for the next unlabel data. 
The model's top-10 predictions are the references for human labeling.
The experiment results shows that, the models trained upon our self-guided dataset can achieve ~90% accuracy in its own training dataset.

#### Keywords
Deep Learning, Human Labeling, Data Centric, Human-In-The-Loop

### 1. Introduction

Deep Learning has been shown to be effective for many artificial intelligence tasks. 
In human-labeled industry deep learning application, data quality or label quality determine the model performance.
The data quality should reach 80%-90% accuracy for model at least, in order to apply the re-label method \cite{ref1} to further improve to more than 95% accuracy. 


### 2. Our Method

![fig1](/assets/png/self-reference/fig1.png)

#### 2.1 Seed data prepare 
The seed 10% data should uniformly sample from all the 100% data. Take text related task as example, we first sort the text dataset by string order and then sample by `1 % 10`. The seed 10% data's reference labels can be predicted by rule/pattern.

#### 2.2 



### 3. Experiments


### 4. Analysis

### 5. Relate Works

### 6. Conclusion


### References
```
\bibitem{ref1}
Guo, Tong (2021): Re-Label Is All You Need. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.17128475.v8 
```
