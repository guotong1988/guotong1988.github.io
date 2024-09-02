---
layout: post
title: "Simple Self-Eval-Drop for Leveraging LLMs"
date: 2024-09-01
category: research
author: "Tong Guo"
description: "Simple Self-Eval-Drop for Leveraging LLMs"
---
# Simple Self-Eval-Drop for Leveraging LLMs

### Abstract

Leveraging large language models (LLMs) to directly inference or build training datasets has becoming an important method for natural language processing (NLP) applications.
However the quality of the training datasets or inference results of LLMs exist some noise data. In this paper, we propose our self-eval-drop method to clean the datasets.
We first sort the to-clean dataset and then sample a seed dataset from the to-clean dataset. Then we manually evaluate the seed dataset and find badcases in the seed dataset.
Then we loop the badcases to find the (batch) similar data in the to-clean dataset for each of the badcases. The manually evaluation results show that our simple self-eval-drop method can improve the accuracy of the to-clean dataset to more than 97% by only sampling and evaluating about 1% of the to-clean dataset as the seed dataset.

### Introduction

In recent years, the development of large language models (LLMs) has brought breakthroughs on NLP applications.
