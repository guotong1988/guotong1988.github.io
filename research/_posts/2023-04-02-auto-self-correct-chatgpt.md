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

We propose Auto Self-Correct ChatGPT (ASC-GPT) to solve the problem to let human teach ChatGPT to refresh/expand ChatGPT's knowledge automatically. 
The core idea of our method is: Correcting/Teaching the ChatGPT's knowledge equals to editing the related data in the whole train-dataset of ChatGPT's policy model.

## 1. Introduction

Large language models (LLM) like ChatGPT and GPT-4 [1] are extremely powerful in multi-task NLP learning.
LLM also improves the development of continuous learning, which means LLM can learn new data without forgeting the old data.
The problem now is that ChatGPT contains some knowledge that is not aligned to human requirements.

Our work contains these contributions:

1. We try to solve the problem to let human teach ChatGPT, based on the ChatGPT's ability to chat with human.

2. We try to correct the train-dataset of ChatGPT's policy model by ChatGPT itself automatically, while ChatGPT is chating with human.

3. We discuss some possible details to implement our goal.


## 2. Method

### 2.1 Procedures

#### 2.1.1 Train Policy Model 

The procedures for training policy model is shown in Fig 1. 

![fig1](/assets/png/self-correct-chatgpt/fig1.png)

In Fig 1. the three steps are:

Step-1. This step is same to InstructGPT's [4] Step-1. In this step, we use the init train-dataset to train the first policy model, based on the pre-trained language model.

Step-2. This step is different to InstructGPT's Step-2. This step do not have the reward part of InstructGPT, because the high quality data is the most important. The high quality model's outputs are selected by human and merged to the init train-dataset.

Step-3. Using the train-dataset self-predict method [3] to find the potential wrong/noise data for human to check and fix. Then we get a better train-dataset to train a better policy model. This step can be done for several times.

#### 2.1.2 Teach Policy Model 

The procedures for teaching policy model new knowledge is shown in Fig 2.

![fig2](/assets/png/self-correct-chatgpt/fig2.png)

In Fig 2. the six steps are:

Step-1. Human inputs a prompt to policy model and find a wrong response. Then human asks policy model to correct.

Step-2. Policy model call the search system to find the related data corresponding to the prompt of Step-1. In this step, policy model transforms natural language to code to call the search system. 

Step-3. The search system fetch the related data from train-dataset and show to human. The search system uses the keyword-based searching algorithm. We first split the prompt-response-pair of Step-1 to sub-strings. Then we search the sub-strings in the train-dataset. 

Step-4. Human confirms and inputs the right data to replace/edit the fetched related data. In this step, the policy model can also recommend some potential ways to fix the data.

Step-5. Policy model processes human's input data to the right format that same to the train-dataset.

Step-6. The new data is merged/replaced into the train-dataset. The policy model is trained again.

### 2.2 Modules

The ASC-GPT contains these sub-modules:

#### 2.2.1 Search System

The keyword-based coded search system that can find the related data in the whole train-dataset of policy model. 
Policy model interacts with this system, following the instructions from human. 
For example, human find a wrong response of policy model. The search system split the corresponding prompt-response-pair to sub-strings. Then the search system takes the sub-strings as queries.
The search system can also be implemented by policy model: We input the prompt that ask policy model to find similar data in its own train-dataset.

#### 2.2.2 Policy Model

We have the chating dataset that can train policy model to chat with human. The policy model can accept these human instructions: 

1. Human instruction that ask policy model to call the search system to find the related data. The policy model transforms natural language to code to call the search system.

2. Human instruction that confirm to do the editing of related data, when human think ChatGPT's response is wrong. 

#### 2.2.3 Data Editing Module

The editing/replacing/adding coded system for fetched related data. 
Policy model interacts with this system, using the knowledge from human and call this system to correct the data in train-dataset.

## 3. Discussion

### Similar Search

We think the model inference equals to finding the similar prompt-response-pair in train-dataset. 
So fixing the model knowledge equals to fixing the its most similar related data.

### Without Reward model 

Our mothod removes the reward model of InstructGPT [4]. Because we think the most important thing is the train-dataset and its quality.


## 4. Related Works

Auto-GPT [2] proposes the goal that attempts to make GPT-4 fully autonomous, which is a great work and do not conflict to our method.
Auto-GPT try to solve the problem that let ChatGPT interact with the internet. Our method focus on allowing human to teach ChatGPT. 

## 5. Conclusion

We propose Auto Self-Correct ChatGPT to solve the problem that allow human to teach ChatGPT/policy-model to refresh/expand ChatGPT's knowledge.
The core idea of our method is: Editing the policy model's knowledge equals to editing the related data in the whole train-dataset of policy model.

## References

```
[1] OpenAI. 2023. Gpt-4 technical report.

[2] https://github.com/Significant-Gravitas/Auto-GPT

[3] Guo T. The Re-Label Method For Data-Centric Machine Learning[J]. arXiv preprint arXiv:2302.04391, 2023.

[4] Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. Advances in Neural Information Processing Systems, 2022, 35: 27730-27744.
```

 
