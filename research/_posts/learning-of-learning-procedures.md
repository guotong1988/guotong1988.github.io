---
layout: post
title: "The Learning of Learning Procedures"
date: 2023-11-01
category: research
author: "Tong Guo"
description: "The Learning of Learning Procedures"
---


# The Learning of Learning Procedures

### Abstract
We propose Learning of Learning Procedures (LLP) to solve the problem to let human teach machine to refresh/expand its knowledge automatically by natural language conversation. LLP is an AI system to solve the problem that let machine to learn knowledge in conversation interacting with human automatically.

The core idea of our method is: Based on large language model (LLM)'s conversation ability, we train an intent recognition model to determine whether the system need to learn new knowledge. 

We propose the procedures to implement our idea and discuss the reasons why we design these procedures.

### 1. Introduction

In recent years, deep learning \cite{ref2} and GPT-based \cite{ref12} model have shown significant improvement on almost all the deep learning tasks. However there is currently a lack of a way for machines to learn from humans in conversations

The human learning procedure is the process of humans interacting with the world to determine whether they should learn new knowledge. For example, when humans are reading books or watching videos,  Human beings will judge what knowledge should be learned into the brain from learning materials. If the learner cannot judge by himself, the human teacher will tell or emphasize the knowledge point.

![table1](/assets/png/llp/table1.png)

![table2](/assets/png/llp/table2.png)

#### 1.1 Problem Definition

The definition of Learning Procedures (LP) in our work is how the system automatically updates/expands its training datasets with human conversations.

In other words, our approach optimizes the automatic expansion and updating of the training datasets.

#### 1.2 Problem Importance

Solving the problem means we give any human the ability to label data for our system.

Solving the problem means our system can improve itself through conversations with users.

### Discussion

#### Limitation
We need to define all possible deep learning tasks in advance.

The main purpose of our approach is to give experts a natural language interface to build a task-oriented dialogue system, based on our LLP system.

### Conclusion



