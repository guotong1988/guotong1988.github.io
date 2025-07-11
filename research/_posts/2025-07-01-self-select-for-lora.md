---
layout: post
title: "Self-Predict And Manual-Select For Improving LoRA-based Domain Fine-tuning"
date: 2025-07-01
category: research
author: "Tong Guo"
description: ""
---
# Self-Predict And Manual-Select For Improving LoRA-based Domain Fine-tuning
### Abstract

LoRA fine-tuning preserves the information of the base LLMs
while incorporating domain-specific data through fine-tuning. 
Therefore, if we use QA-pair domain training dataset to LoRA fine-tune a LLM
and then employ this fine-tuned LLM to predict the domain training dataset itself, 
we can prepare two or more answers for each QA-pair's question. 
We manually label the optimal answer from the answers, replace the original answer, 
and proceed to the next round of LoRA fine-tuning.
Thus, we can continuously optimize the training dataset through iterative self-predict and human-select.
This method can also be applied to multi-turn QA fine-tuning datasets.
The human evaluation results of the fine-tuned LLM demonstrate that our approach is effective.
