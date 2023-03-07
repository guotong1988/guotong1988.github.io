---
layout: post
title: "Learning From How Children Learning"
date: 2023-03-01
category: research
author: "Tong Guo"
description: "Learning From How Children Learning"
---


# Learning From How Children Learning

### Abstract

The learning procedure of human child are:

Step-1, the child observes how adults do, which we view as the initial supervised training dataset for model. 

Step-2, the child interacts with the physical world and get feedbacks like can-do/cannot-do labels by adults, which means the model predict for the test dataset, then the predicted labels become ground-truth labels after humans' agreements. 

Step-3, the child updates the knowledge in brain, which means the test dataset with ground-truth labels is merged into the initial training dataset to update the model policy. 

Step-4, the brain-updated child interacts with the world get feedbacks again, and back to Step-2 and loop again.

### 1. Introduction

In previous works, we focus on re-label the noisy data in initial training dataset. In this paper, we focus on solving the problem of getting better results that are summarized from pre-trained GPT.
