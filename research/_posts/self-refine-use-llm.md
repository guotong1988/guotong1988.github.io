---
layout: post
title: "Self-Refine Learning For Leveraging LLMs Data"
date: 2024-12-31
category: research
author: "Tong Guo"
description: "Self-Refine Learning For Leveraging LLMs Data"
---
# Self-Refine Learning For Leveraging LLMs Data

### Abstract

In industry NLP application, our data by prompting large language models (LLMs) has a certain number of noise data. 
We present a simple method to find the noise data and remove them. We select the noise data whose LLMs' label is not contained in the top-K model's predictions. 
The experiment result shows that our method works. For industry deep learning application, our method improves the text classification accuracy from % to % in dev dataset, 
and improve the human-evaluation accuracy from % to %. 
The conclusion is: The self-predict and remove method of this paper can improve the accuracy to about 90\% automatically, if the base dev accuracy is around 80\%.
