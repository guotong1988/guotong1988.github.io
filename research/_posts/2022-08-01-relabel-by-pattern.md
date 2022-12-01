---
layout: post
title: "Re-Label By Data Pattern Is All You Need For Knowledge Driven Deep Learning"
date: 2022-08-01
category: research
comments: true
author: "Tong Guo"
description: "Re-Label By Data Pattern Is All You Need For Knowledge Driven Deep Learning"
---


# Re-Label By Data Pattern Is All You Need For Knowledge Driven Deep Learning

### abstract
In industry deep learning application, we should fix the badcase by human evaluation after we achieve more than 95% accuracy at dev dataset. The badcase reason is from the wrong rule/knowledge of human labeling and will cause low accuracy under human evaluation. In this paper, we propose the pattern-based method to fix the badcase for industry application inference. We propose the pipeline to solve the problem and improve the accuracy of human evaluation. The experiment results verify our idea, which means label-edit is the method to implement controllable deep learning application.

#### keywords

Deep Learning, Pattern Recognition, Fuzzy Matching, Similar Search

### Introduction

In industry deep learning \cite{ref1} application, we use the re-label method\cite{ref2} to achieve more than 95% accuracy at dev dataset. But under human evaluation for industry application inference, the accuracy is only around 90%-94%. The reason for this problem can be found on the badcase review. In the badcase review, we find that the badcase is caused by wrong rule/knowledge of human labeling. In order to fix the badcase of training data. we propose the pattern-based re-label pipeline. The intuition is that the badcase prediction data should contain the same pattern in the training data. Take text classification as example, it means the badcase prediction data contains the same tokens in the training data. So we could retrieve all the training text which contains the same pattern/tokens and re-label them.

In this paper, knowledge-driven means the knowledge for labeling when we teach the human how to label. Our method is a kind of human-in-the-loop which means we refresh the knowledge for human to re-label the data. It means we implement controllable deep learning application by label-edit.

### Methods

In this section, we propose the pipeline to improve the human evaluation accuracy.

![](/assets/png/relabel-by-pattern/fig1.png)

#### Step-1
Based on the human-labeled training dataset, we train a deep model and predict for industry application dataset. Human evaluate the current deep model prediction dataset and find the badcase patterns. Take text classification task as example, the pattern is the tokens from the badcase prediction text. For computer vision task, the pattern is the pixel token from the badcase prediction image.

If there are no training data that is recognised by badcase patterns, we newly label the real application data that is recognised by badcase patterns and merge to the training data. Also we get more than 95% accuracy in dev dataset by \cite{ref6} method.


#### Step-2
Re-set the labeling rule/knowledge for human. Then the knowledge-refreshed human re-label the training data that is recognised by badcase patterns of Step-1. Then we train to get a new version of current deep model.


#### Loop Step-1 and Step-2

Back to last Step-1 and re-evaluate the last Step-2 model's predictions and find the new badcase patterns.  



### Experimental Results
In the section, we describe the experiment results. Take our text classification task as example. The result is shown in Table 1. In Table 1, data-v1 means the origin training data. And data-v2 and data-v3 means the training data after we use our method for 2 loops, based on data-v1.

![](/assets/png/relabel-by-pattern/table1.png)

### Related Works
There are works that inject knowledge into deep model. But they are different from our work. They focus on model-centric part, but our method focuses on the human-in-the-loop labeling data-centric part.

#### Rule-enhanced Text2Sql
In the text2sql task, there is work\cite{ref3} that injects external entity type-info/knowledge into deep model. There is work \cite{ref4} that inject database design rules/knowledge into deep model. These works give external information that can not be captured by an end-to-end deep model.

#### Keyword-enhanced NER
In the named-entity-recognise task, there is work\cite{ref5} that injects the entity dictionary into deep model. This work give more information for the deep model if the entities of training data is similar to test data.

### Conclusion

In this paper, we solve the problem of the low accuracy under human evaluation after we achieve good accuracy of dev dataset. We propose the pipeline: First, we find the badcase in the prediction data and summarize the right knowledge for human labeling. Second, we summarize the pattern from prediction badcase data and re-label the training data which match the pattern. Loop this pipeline means we inject the right knowledge/rule to the deep model by re-label the training data retrieved by pattern from badcase prediction data. The experiment results verify our idea.
Our idea can apply to a broad set of deep learning industry applications.

### References
```
\bibitem{ref1}
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25: 1097-1105.

\bibitem{ref2}
Guo T. Learning From Human Correction For Data-Centric Deep Learning[J]. arXiv preprint arXiv:2102.00225, 2021.

\bibitem{ref3}
Yu T, Li Z, Zhang Z, et al. Typesql: Knowledge-based type-aware neural text-to-sql generation[J]. arXiv preprint arXiv:1804.09769, 2018.

\bibitem{ref4}
Guo T, Gao H. Content enhanced bert-based text-to-sql generation[J]. arXiv preprint arXiv:1910.07179, 2019.

\bibitem{ref5}
https://github.com/guotong1988/Chinese-NER-InjectDictRule

\bibitem{ref6}
Guo, Tong (2021): Re-Label Is All You Need. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.17128475.v5 
```