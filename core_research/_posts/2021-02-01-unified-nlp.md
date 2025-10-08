---
layout: post
title: "A Unified Framework for NLP Tasks by ReLabel Method"
date: 2025-10-01
category: core_research
author: "Tong Guo"
description: "A Unified Framework for NLP Tasks by ReLabel Method"
---


# A Unified Framework for NLP Tasks by ReLabel Method

### Abstract
In industry deep learning application, our dataset has a certain number of noisy data. The init datasets are from human labeling or LLM (large language model) generation or user behavior log.
To solve this problem and achieve more than 90 score in dev dataset, we present a framework to find the noisy data and relabel the noisy data, 
given the model predictions as references in relabeling. The process of relabeling can be done manually or using LLM for annotation.
In this paper, we illustrate our idea for a broad set of deep learning tasks, includes classification, sequence tagging, object detection, sequence generation, 
click-through rate prediction. The dev dataset evaluation results and human evaluation results verify our idea.

#### Keywords
NLP, LLM

### 1. Introduction

In recent years, deep learning \cite{ref1} model and LLM have shown significant improvement on natural language processing(NLP), 
computer vision and speech processing technologies. However, the model performance is limited by the dataset quality. 
The main reason is that the dataset has a certain number of noisy data. 
In this paper, we present a framework to find the noisy data and relabel the noisy data, 
then we further illustrate our idea for sequence tagging, object detection, sequence generation, click-through rate (CTR) prediction.

### 2. Method
