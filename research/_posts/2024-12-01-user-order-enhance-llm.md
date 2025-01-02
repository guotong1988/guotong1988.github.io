---
layout: post
title: "Leveraging User Behaviour to Enhance LLMs for Query Analysis in E-commerce"
date: 2024-12-01
category: research
author: "Tong Guo"
description: "Leveraging User Behaviour to Enhance LLMs for Query Analysis in E-commerce"
---
# Leveraging User Behaviour to Enhance LLMs for Query Analysis in E-commerce

### Abstract

The advantage of using LLMs (Large Language Models) for query understanding is that we do not need to manually annotate data. But the problem with using LLMs for query analysis in e-commerce search is that the accuracy rate can at most reach up to 88%-90%, due to the lack of domain specific information. Meanwhile, based on mining query tags from user orders, the accuracy can reach over 95%. But the coverage rate of order-based query tags is ～50%, which is limited by the amount of user behavior logs, especially when the business scope is small. In this article, we propose a method for constructing training dataset for query understanding/tags, which combines the data by LLMs with the order-based data. Models trained on this dataset can achieve a coverage rate of over 95% and an accuracy rate of over 95% for query tagging.

### 1. Introduction

Query analysis is the task of query tagging, such as query synonyms, query categories, etc.

### 2. Method
After cleaning the dataset by prompting the LLMs, combine it with the dataset mined based on user orders to obtain the final training dataset. The LLMs dataset and order-based dataset are constructed as follows:

#### 2.1 Prompt LLMs for Query Analysis
![fig1](/assets/png/self-eval-drop/fig1.png)
#### 2.2 User Behaviour Based Query Analysis
Count the item tags with the highest order amount within the same search query.
![fig1](/assets/png/user-order-enhance-llm/fig1.png)
Count the search querys with the highest and second highest amount of orders in the same item.
![fig2](/assets/png/user-order-enhance-llm/fig2.png)
#### 2.3 Use Merged Dataset To Train 
The final merged dataset has sufficient data quantity and quality to train a generative model that outputs tags for a given input query. Specifically, we used T5. This T5 model performs tagging predictions for each query.

### 3. Evaluation

Our T5 model trained on the final dataset achieves a coverage rate of over 95% and an accuracy rate of over 95% for query tagging.

### 4. Discussion

There are many ways to merge the dataset from LLMs and the dataset based on orders:

1) Directly mix the order-based dataset and the cleaned LLMs dataset, if the accuracy of the LLMs dataset has also reached over 95%, considering the dataset based on orders already has an accuracy of over 95%.

2) If the prediction results for each query from the T5 model trained on order-based dataset are equal to those prediction from the LLMs, then append the training dataset with this data. If there is insufficient labeling manpower to perform the aforementioned LLMs data cleaning manually.


### 5. Conclusion
This paper proposes a method of constructing a training dataset using query tag mining based on user orders, while also integrating a dataset for query tagging from LLMs. The T5 model trained on the final dataset can achieve 95% coverage and 95% accuracy in query understanding.

### Reference
```

\bibitem{ref1}

\bibitem{ref2}

\bibitem{ref3}

\bibitem{ref4}

\bibitem{ref5}

```
