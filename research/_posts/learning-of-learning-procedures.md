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
We propose Learning of Learning Procedures (LLP) to solve the problem to let human teach machine to refresh/expand its knowledge automatically by natural language conversation. LLP is an AI system to solve the problem that let machine to learn knowledge in conversation interacting with human automatically.

The core idea of our method is: The learning of a machine means updating its training datasets. Based on large language model (LLM)'s conversation ability, we train an addtional intent recognition model to determine whether the system need to learn new knowledge. 

It is critical, but except for the developers of the AI system, others have no interface to do this education for machine. Also, based on our LLP system, our approach can give users a natural language interface to build their own task-oriented dialogue system, question answering system, or reasoning system.

We propose the procedures to implement our idea and discuss the reasons why we design these procedures.

### 1. Introduction

In recent years, deep learning \cite{ref2} and GPT-based \cite{ref12} model have shown significant improvement on almost all the deep learning tasks. However there is currently a lack of a way for machines to learn from humans in conversations

The human learning procedure is the process of humans interacting with the world to determine whether they should learn new knowledge. For example, when humans are reading books or watching videos,  Human beings will judge what knowledge should be learned into the brain from learning materials. If the learner cannot judge by himself, the human teacher will tell or emphasize the knowledge point.

![table1](/assets/png/llp/table1.png)

![table2](/assets/png/llp/table2.png)

#### 1.1 Problem Definition

The definition of Learning Procedures (LP) in our work is how the system automatically updates/expands its training datasets with human conversations.

In other words, our approach optimizes the automatic expansion and updating of the training datasets.

#### 1.2 Problem Importance

Solving the problem means we give any human the ability to label data for our system.

Solving the problem means our system can improve itself through conversations with users.

Based on our LLP system, our approach can give users a natural language interface to build their own task-oriented dialogue system, question answering system, or reasoning system.

### Discussion


#### Limitation

We need to define all possible deep learning tasks in advance.



### References
```
\bibitem{ref1}
Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

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

### Conclusion



