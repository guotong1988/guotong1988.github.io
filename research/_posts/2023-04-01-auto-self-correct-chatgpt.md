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

Large language models (LLM) like ChatGPT and GPT-4 [17] are extremely powerful in multi-task NLP learning.
LLM also improve the development of continuous learning, which means LLM can learn new data without forgeting the old data.


### Method

The ASC-GPT contains these sub-modules:

1, The keyword-based coded search system that can find the related data in the whole fine-tune train-dataset.

2, The chat dataset that can train ChatGPT to chat with human, which can accept these human instructions: 

2.1, Instruction that ask ChatGPT to call the search system to find the related data. 

2.2, Instruction that confirm to do the correction of related data, when human think ChatGPT is wrong.

3, The editing/replacing module for related data, which is also a coded system.

### Conclusion


### References

```
[1] OpenAI. 2023. Gpt-4 technical report.



```

 
