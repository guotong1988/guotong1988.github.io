---
layout: post
title: "Automatic Label Error Correction"
date: 2024-02-01
category: research
author: "Tong Guo"
description: "Automatic ReLabel Is All You Need For 97% Accuracy At Dev Dataset"
---


# Automatic Label Error Correction

### Abstract

In the previous works, we verify that manual re-labeling the training data that model prediction do not equal to the manual label, improve the accuracy at dev dataset and human evaluation.
But manual re-labeling requires a lot of human labor to correct the noisy data in the training dataset. We think that the noisy data is the training data that model prediction and manual label is not equal.
To solve this problem, we propose the randomly re-setting method that automaticly improve the accuracy at dev dataset and human evaluation.
The core idea is that, we randomly re-set all the noisy data to model prediction or the manual label. All the noisy data means the noisy data in training dataset and dev dataset.
We use the highest accuracy of dev dataset in the all possible randomly re-setted dev datasets, to judge the quality of randomly re-setted training datasets.
The motivation of randomly re-setting is that The best dataset must be in all randomly possible datasets.

### Introduction

### Method
