---
layout: post
title: "The Learning of Learning Procedures"
date: 2023-11-01
category: research
author: "Tong Guo"
description: "The Learning of Learning Procedures"
---


# The Learning of Learning Procedures

### Abstract

The current large language models (LLMs) mainly lacks this capability: autonomous learning capability.

We propose Learning of Learning Procedures (LLP) to solve the problem to let human teach machine to refresh/expand its tasks definition automatically by natural language conversation.
LLP system is an AI system that contains many deep learning tasks. LLP system provide the natural language interface to let machine to learn new tasks definition in conversation interacting with human automatically.

It is critical, but except for the developers of the AI system, others have no interface to do this education for machine. Also, based on our LLP system, our approach can give users a natural language interface to build their own task-oriented dialogue system, question answering system, or reasoning system.

We propose the methods to implement our idea and discuss the reasons why we design these methods.

### 1. Introduction

The human learning procedure is the process of humans interacting with the world to determine whether they should learn new knowledge. For example, when humans are reading books or watching videos,  Human beings will judge what knowledge should be learned into the brain from learning materials. If the learner cannot judge by himself, the human teacher will tell or emphasize the knowledge point.

In recent years, deep learning (DL) \cite{ref2} and GPT-based \cite{ref12} model have shown significant improvement on almost all the DL tasks. However there is currently a lack of a way for machines to learn automatically from humans in conversations.

Currently, there are two ways to solve the problem of autonomous learning \cite{ref1} based on the DL framework: One is to add data under existing reasoning tasks, and the other is to define a new DL task that meets the requirements. Our work is to optimize the automatic expansion of DL task in the AI system.

\cite{ref1} has a complete definition of machine reasoning, which defines machine reasoning as a standard DL task. Based on the current DL framework, defining the task of the reasoning problem and adding it with the corresponding data is equivalent to solving this reasoning problem by DL.

This work mainly solves the problem of automatic definition tasks. For solutions to automatically update data, please refer to the appendix.

#### 1.1 Problem Definition

The definition of Learning Procedures (LP) in our work is the DL tasks in the AI system.

The definition of Learning of Learning Procedures (LLP) in our work is how the AI system automatically updates/expands its DL tasks definition with human conversations.

#### 1.2 Problem Importance

Solving this problem means we give people the ability to create new DL tasks (and then add new labeled data) for the AI system, except the developers of the AI system.

Based on our LLP system, our approach can give users a natural language interface to build their own task-oriented dialogue system, question answering system, or reasoning system.

### The Learning Of Task Definition

![table2](/assets/png/llp/table3.png)

### Discussion


### Conclusion


### References
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

#### Add New Data To Existing Tasks

The core idea of updating the datasets is: The learning of a existing task equals to updating its training datasets. Based on large language model (LLM)'s conversation ability, we train an additional intent recognition model to determine whether the system need to learn new knowledge. 

![table1](/assets/png/llp/table1.png)

![table2](/assets/png/llp/table2.png)

