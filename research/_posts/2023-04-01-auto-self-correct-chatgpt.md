---
layout: post
title: "Auto Self-Correct ChatGPT"
date: 2023-03-01
category: research
author: "Tong Guo"
description: "Auto Self-Correct ChatGPT"
---
# Auto Self-Correct ChatGPT

### Abstract

We propose Auto Self-Correct ChatGPT (ASC-GPT) to solve the problem that allow human to teach ChatGPT to refresh its knowledge.

Correcting the ChatGPT knowledge, means correcting the related data in the whole training dataset of ChatGPT.


### Method

The ASC-GPT contains these sub-modules:

1, The keyword-based coded search system that can find the related data in the whole training dataset.

2, The chat dataset that can train ChatGPT to chat with human, which can accept these human instructions: 

2.1, Instruction that ask ChatGPT to call the search system to find the related data. 

2.2, Instruction that confirm to do the correction of related data, when human think ChatGPT is wrong.

3, The editing/replacing module for related data, which is also a coded system.

 
