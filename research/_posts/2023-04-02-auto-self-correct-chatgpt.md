---
layout: post
title: "Auto Self-Correct ChatGPT"
date: 2023-04-02
category: research
author: "Tong Guo"
description: "Auto Self-Correct ChatGPT"
---
# Auto Self-Correct ChatGPT

## Abstract

We propose Auto Self-Correct ChatGPT (ASC-GPT) to solve the problem that allow human to teach ChatGPT to refresh/expand ChatGPT's knowledge automatically. 
The core idea of our method is: Correcting/Teaching the ChatGPT's knowledge equals to editing the related data in the whole train-dataset of ChatGPT's policy model.

## 1. Introduction

Large language models (LLM) like ChatGPT and GPT-4 [1] are extremely powerful in multi-task NLP learning.
LLM also improves the development of continuous learning, which means LLM can learn new data without forgeting the old data.
The problem now is that ChatGPT contains some knowledge that is not aligned to human requirements.
Our work contains these contributions:

1. We try to solve the problem: We let human to teach ChatGPT, based on the ChatGPT's ability to chat with human.

2. We try to correct the train-dataset of ChatGPT's policy model by ChatGPT itself automatically, while ChatGPT is chating with human.

3. We discuss some possible details to implement our goal.


## 2. Method

### 2.1 Procedures

#### 2.1.1 Train Policy Model 

The procedures for training policy model is shown in Fig 1. 

![fig1](/assets/png/self-correct-chatgpt/fig1.png)

In Fig 1. the three steps are:

Step-1. This step is same to InstructGPT's [4] Step-1. In this step, we use the init train-dataset to train the first policy model.

Step-2. This step is different to InstructGPT's step-2. This step do not have the reward part of InstructGPT. The right model outputs are selected by human and merged to the init train-dataset.

Step-3. Using the train-dataset self-predict method [3] to find the potential wrong data for human to check and fix. Then we get a better train-dataset and a better policy model.

#### 2.1.2 Teaching Policy Model 

The procedures for teaching policy model new knowledge is shown in Fig 2.

![fig2](/assets/png/self-correct-chatgpt/fig2.png)

In Fig 2. the six steps are:

Step-1. Human chats with policy model and find something wrong and ask policy model to correct.

Step-2. Policy model call the search system to find the related data corresponding to the prompt of Step-1.

Step-3. The search system fetch the related data from train-dataset and show to human.

Step-4. Human confirms and inputs the right data to replace/edit the fetched related data.

Step-5. Policy model processes human's input data to the right format that same to the train-dataset.

Step-6. The new data is merged/replaced into the train-dataset. The policy model is trained again.

### 2.2 Modules

The ASC-GPT contains these sub-modules:

#### 2.2.1 Similar Text Search System

The keyword-based coded search system that can find the related data in the whole train-dataset of policy model. 
ChatGPT interacts with this system, following the instructions from human.

#### 2.2.2 The Ability To Chat With Human

We have the chating dataset that can train ChatGPT to chat with human, which can accept these human instructions: 

1. Human instruction that ask ChatGPT to call the search system to find the related data. 

2. Human instruction that confirm to do the editing of related data, when human think ChatGPT's response is wrong. 

#### 2.2.3 The Data Editing Module

The editing/replacing/adding coded system for fetched related data. 
ChatGPT interacts with this system, using the knowledge from human to correct data.




## 3. Related Works

Auto-GPT [2] proposes the goal that attempts to make GPT-4 fully autonomous, which is a great work and do not conflict to our method.
Auto-GPT try to solve the problem that let ChatGPT interact with the internet. Our method focus on allowing human to teach ChatGPT. 

## 4. Conclusion

We propose Auto Self-Correct ChatGPT to solve the problem that allow human to teach ChatGPT/policy-model to refresh/expand ChatGPT's knowledge.
The core idea of our method is: Editing the policy model's knowledge equals to editing the related data in the whole train-dataset of policy model.

## References

```
[1] OpenAI. 2023. Gpt-4 technical report.

[2] https://github.com/Significant-Gravitas/Auto-GPT

[3] Guo T. The Re-Label Method For Data-Centric Machine Learning[J]. arXiv preprint arXiv:2302.04391, 2023.

[4] Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. Advances in Neural Information Processing Systems, 2022, 35: 27730-27744.
```

 
