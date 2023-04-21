---
layout: post
title: "Auto Self-Correct ChatGPT"
date: 2023-04-01
category: research
author: "Tong Guo"
description: "Auto Self-Correct ChatGPT"
---
# Auto Self-Correct ChatGPT

### Abstract

We propose Auto Self-Correct ChatGPT (ASC-GPT) to solve the problem that allow human to teach ChatGPT to refresh its knowledge. Correcting the ChatGPT knowledge, means correcting the related data in the whole fine-tune train-dataset of ChatGPT.

### Introduction

Large language models (LLM) like ChatGPT and GPT-4 [1] are extremely powerful in multi-task NLP learning.
LLM also improves the development of continuous learning, which means LLM can learn new data without forgeting the old data.
Auto-GPT proposes the goal that attempts to make GPT-4 fully autonomous.
The problem now is that ChatGPT contains some knowledge that is not aligned to human requirements.
Our work contains these contributions:

1, We try to solve the problem: We let human to teach ChatGPT, based on the ChatGPT's ability to interact with human.

2, We try to correct the fine-tune train-dataset of ChatGPT by ChatGPT itself automatically, while ChatGPT is interacting with human.

2, We discuss some possible details to implement our goal.

### Method

#### Procedures

The whole procedure is shown in Fig 1.

![fig1](/assets/png/self-correct-chatgpt/fig1.png)

#### Modules

The ASC-GPT contains these sub-modules:

##### Similar Text Search System

The keyword-based coded search system that can find the related data in the whole fine-tune train-dataset.

##### The Ability To Chat With Human

The chat dataset that can train ChatGPT to chat with human, which can accept these human instructions: 

1, Instruction that ask ChatGPT to call the search system to find the related data. 

2, Instruction that confirm to do the correction of related data, when human think ChatGPT is wrong.

##### The Data Editing Module

The editing/replacing module for fetched related data, which is also a coded system.

### Conclusion


### References

```
[1] OpenAI. 2023. Gpt-4 technical report.

[2] https://github.com/Significant-Gravitas/Auto-GPT


```

 
