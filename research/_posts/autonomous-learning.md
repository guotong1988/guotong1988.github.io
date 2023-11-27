---
layout: post
title: "Building Student Module for Large Language Models"
date: 2023-11-01
category: research
author: "Tong Guo"
description: "Towards Autonomous Learning of Large Language Models"
---


# Building Student Module for Large Language Models

## Abstract

The current large language models (LLMs) mainly lacks this capability: autonomous learning capability. Also, we can not prepare all the data/knowledge in the world for LLMs.

We propose Learning In Conversation (LIC) to solve the problem to let human teach machine to refresh/expand its data/knowledge automatically by natural language conversation.
LIC system is an AI system that contains many deep learning tasks. LIC system provide the natural language interface to let machine to learn new data/knowledge in conversation interacting with human automatically.

It is critical, but except for the developers of the AI system, others have no natural language interface to do this education for machine. 

Based on large language models (LLMs) conversation ability, we train an additional intent recognition model to determine when the AI system need to learn new data/knowledge. We add a module for editing the training dataset of LLMs. 

We propose the methods to implement our idea and discuss the reasons why we design these methods.

## 1. Introduction

The human learning procedure is the process of humans interacting with the world to determine whether they should learn new knowledge. For example, when humans are reading books or watching videos,  Human beings will judge what knowledge should be learned into the brain from learning materials. If the learner cannot judge by himself, the human teacher will tell or emphasize the knowledge point.

In recent years, deep learning (DL) \cite{ref2} and GPT-based \cite{ref12} model have shown significant improvement on almost all the DL tasks. However there is currently a lack of a way for machines to learn automatically from humans in conversations. And we can not prepare all the data/knowledge in the world for LLMs.
 

In this paper we propose a framework of AI system, called LIC system. LIC system is based on LLMs and have the ability to learn in the conversation with human.

The LIC system has these modules: 

1) LLMs to do conversation with human.
2) An intent recognition module to determine when to update data/knowledge in the system.
3) An data management module to update the training dataset.

## 2. The AI system

![fig1](/assets/png/auto-learn/fig1.png)

#### 2.1 Common Conversation Module

#### 2.2 Intent Recognition Module

![table1](/assets/png/auto-learn/table1.png)


![table2](/assets/png/auto-learn/table2.png)

## 4. Discussion

Under the deep learning framework, all data/knowledge is first memorized by the model. 

## 5. Related Works

## 6. Conclusion


## References
```
\bibitem{ref1}
Yu F, Zhang H, Wang B. Nature language reasoning, a survey[J]. arXiv preprint arXiv:2303.14725, 2023.

\bibitem{ref2}
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25: 1097-1105.

\bibitem{ref3}
Wang A, Singh A, Michael J, et al. GLUE: A multi-task benchmark and analysis platform for natural language understanding[J]. arXiv preprint arXiv:1804.07461, 2018.

\bibitem{ref4}
Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014.

\bibitem{ref5}
Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting[J]. The journal of machine learning research, 2014, 15(1): 1929-1958.

\bibitem{ref6}
Xie Q, Dai Z, Hovy E, et al. Unsupervised data augmentation for consistency training[J]. arXiv preprint arXiv:1904.12848, 2019.

\bibitem{ref7}
Berthelot D, Carlini N, Goodfellow I, et al. Mixmatch: A holistic approach to semi-supervised learning[J]. arXiv preprint arXiv:1905.02249, 2019.

\bibitem{ref8}
Sohn K, Berthelot D, Li C L, et al. Fixmatch: Simplifying semi-supervised learning with consistency and confidence[J]. arXiv preprint arXiv:2001.07685, 2020.

\bibitem{ref9}
Berthelot D, Carlini N, Cubuk E D, et al. Remixmatch: Semi-supervised learning with distribution alignment and augmentation anchoring[J]. arXiv preprint arXiv:1911.09785, 2019.

\bibitem{ref10}
Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.

\bibitem{ref11}
Northcutt C G, Wu T, Chuang I L. Learning with confident examples: Rank pruning for robust classification with noisy labels[J]. arXiv preprint arXiv:1705.01936, 2017.

\bibitem{ref12}
Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. arXiv preprint arXiv:2203.02155, 2022.

\bibitem{ref13}
Guo, Tong (2022): Re-Label By Data Pattern For Knowledge Driven Deep Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.20485917.v7

\bibitem{ref14}
Northcutt C, Jiang L, Chuang I. Confident learning: Estimating uncertainty in dataset labels[J]. Journal of Artificial Intelligence Research, 2021, 70: 1373-1411.

\bibitem{ref15}
Zha D, Bhat Z P, Lai K H, et al. Data-centric artificial intelligence: A survey[J]. arXiv preprint arXiv:2303.10158, 2023.
```


### Appendix



