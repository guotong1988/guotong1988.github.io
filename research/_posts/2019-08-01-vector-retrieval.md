---
layout: post
title: "Revisiting Semantic Representation and Tree Search for Similar Question Retrieval"
date: 2019-08-01
category: research
author: "Tong Guo"
description: "Revisiting Semantic Representation and Tree Search for Similar Question Retrieval"
---


# Revisiting Semantic Representation and Tree Search for Similar Question Retrieval



## Abstract

This paper studies the performances of BERT combined with tree structure in short sentence ranking task. 
In retrieval-based question answering system, we retrieve the most similar question of the query question by ranking all the questions in datasets. 
If we want to rank all the sentences by neural rankers, we need to score all the sentence pairs. 
However it consumes large amount of time. 
So we design a specific tree for searching and combine deep model to solve this problem. 
We fine-tune BERT on the training data to get semantic vector or sentence embeddings on the test data. 
We use all the sentence embeddings of test data to build our tree based on k-means and do beam search at predicting time when given a sentence as query. 
We do the experiments on the semantic textual similarity dataset, Quora Question Pairs, and process the dataset for sentence ranking. 
Experimental results show that our methods outperform the strong baseline. 
Our tree accelerate the predicting speed by 500%-1000% without losing too much ranking accuracy.
**But TF-IDF and BM25 are still the best methods for information retrieval**

#### Keywords
Information Retrieval, Vector Representation

## 1. Introduction

In retrieval-based question answering system \cite{wang2017bilateral,liu2018finding,guo2019deep}, we retrieve the answer or similar question from a large question-answer pairs. We compute the semantic similar score between question-question pairs or compute the semantic related score of question-answer pairs and then rank them to find the best answer. In this paper we discuss the similar question retrieval. For the similar question retrieval problem, when given a new question in predicting, we get the most similar question in the large question-answer pairs by ranking, then we can return the corresponding answer. We consider this problem as a short sentence ranking problem based on sentence semantic matching, which is also a kind of information retrieval task. 

Neural information retrieval has developed in several ways to solve this problem. This task is considered to be solved in two step: A fast algorithm like TF-IDF or BM25 to retrieve about tens to hundreds candidate similar questions and then the second step leverage the neural rankers to re-rank the candidate questions by computing the question-question pairs similarity scores. So one weakness of this framework with two steps above is that if the first fast retrieval step fails to get the right similar questions, the second re-rank step is useless. So one way to solve this weakness is to score all the question-question pairs by the neural rankers, however it consumes large amount of time. A full ranking may take several hours. See Fig 1. for the pipeline illustration.

![fig1](/assets/png/vector-retrieval/fig1.png)


In this paper, to get the absolute most similar question on all the questions and solve the problem of long time for ranking all the data, inspired by the idea of \cite{zhu2018learning} and \cite{zhu2019joint}, we propose two methods: One is to compute all the semantic vector for all the sentence by the neural ranker offline. And then we encode the new question by the neural ranker online. Tree is an efficient structure for reducing the search space\cite{silver2016mastering}. To accelerate the speed without losing the ranking accuracy we build a tree by k-means for vector distance computation. Previous research \cite{qiao2019understanding,xu2019passage} shows that origin BERT\cite{devlin2018bert} can not output good sentence embeddings, so we design the cosine-based loss and the fine-tune architecture of BERT to get better sentence embeddings. Another method is to compute the similarity score by deep model during tree searching. In this paper, the words, distributed representations and sentence embeddings and semantic vector, are all means the final output of the representation-based deep model.

In summary our paper has three contributions: First, We fine-tuning BERT and get better sentence embeddings, as the origin embeddings from BERT is bad. Second, To accelerate the predicting speed, we build a specific tree to search on all the embeddings of test data and outperform the baseline. Third, after we build the tree by k-means, we search on the tree while computing the similarity score by interaction-based model and get reasonable results.


## 2. Related Works

In recent years, neural information retrieval and neural question answering research has developed several effective ways to improve ranking accuracy. Interaction-based neural rankers match query and document pair using attention-based deep model; representation-based neural rankers output sentence representations and using cosine distance to score the sentence pairs. There are many effective representation-based model include DSSM\cite{huang2013learning}, CLSM \cite{shen2014latent} and LSTM-RNN \cite{palangi2016deep} and many effective interaction-based model include DRMM\cite{guo2016deep} Match-SRNN\cite{wan2016match} and BERT\cite{devlin2018bert}.

Sentence embeddings is an important topic in this research area. Skip-Thought\cite{kiros2015skip} input one sentence to predict its previous and next sentence. InferSent\cite{conneau2017supervised} outperforms Skip-Thought. \cite{arora2016simple} is the method that use unsupervised word vectors\cite{pennington2014glove} to construct the sentence vectors which is a strong baseline. Universal Sentence Encoder \cite{cer2018universal} present two models for producing sentence embeddings that demonstrate good transfer to a number of other of other NLP tasks.

BERT is a very deep transformer-based\cite{vaswani2017attention} model. It first pre-train on very large corpus using the mask language model loss and the next-sentence loss. And then we could fine-tune the model on a variety of specific tasks like text classification, text matching and natural language inference and set new state-of-the-art performance on
them. However BERT is a very large model, the inference time is too long to rank all the sentence. 

We follow the BERT convention of data input format for encoding the natural language question. For single sentence classification task, the question $Q = \{w_1,w_2,...,w_n\}$ is encoded as following:

$$
[CLS], w_1, w_2, ..., w_n, [SEP]
$$


For sentence pair classification task, BERT passes two sentences to the transformer network and the target value is predicted. The question 1 $Q_1 = \{w_1,w_2,...,w_n\}$ and question 2 $Q_2 = \{w_1,w_2,...,w_m\}$ are encoded as following:

$$
[CLS], w_1, ..., w_n, [SEP], w_1, ..., w_m, [SEP] 
$$

where [CLS] is a special symbol added in front of every input example, [SEP] is a special separator token, $n$, $m$ is the token number. Our fine-tune training follows the single sentence classification task convention for representation-based methods and follows the sentence pair classification task convention for interaction-based methods.

## 3. Problem Statement and Approach

### 3.1 Problem Statement 
In this section, we illustrate the short sentence ranking task. In training time, we have a set of question pairs label by 1 for similar and by 0 for not similar. Our goal is to learn a classifier which is able to precisely predict whether the question pair is similar. But we can not follow the same way as sentence pair classification task of BERT, if we want to output the sentence embeddings for each of the sentence. In predicting time, we have a set of questions $Q = \{{q_1,q_2,...,q_n}\}$ that each have a labeled most similar question in the same set $Q$. Our goal is to use a question from the question set $Q$ as query and find the top N similar questions from the question set $Q$. Although the most similar question for the query is the one that we consider to be the most important one in question answering system, but the top N results may be applied to the scenario such as similar question recommendation. In the next section we describe our deep model and the tree building methods to solve this problem.

![fig2-3](/assets/png/vector-retrieval/fig23.png)


### 3.2 Fine-tune Training
In this subsection we describe our fine-tune methods for BERT. We call it representation-based method which fine-tune BERT to get sentence embeddings. 
We call it interaction-based method which fine-tune BERT to compute similarity score of sentence pairs during tree searching.

#### 3.2.1 Representation-based method
 The sketch view is shown in Fig. 2. We input the two questions to the same BERT without concatenate them and output two vector representation. We adds a pooling operation to the output of BERT to derive a fixed sized sentence embedding. In detail, we use three ways to get the fixed sized representation from BERT: 

1. The output of the [CLS] token. We use the output vector of the [CLS] token of BERT for the two input questions. 

2. The mean pooling strategy. We compute mean of all output vectors of the BERT last layer and use it as the representation.

3. The max pooling strategy. We take the max value of the output vectors of the BERT last layer and use it as the representation.

Then the two output vectors from BERT compute the cosine distance as the input for mean square error loss:

$$ 
loss = MSE(u \cdot v / (||u||*||v||),y) 
$$

where $u$ and $v$ is the two vectors and $y$ is the label. The full algorithm is shown in Algorithm 1.

#### 3.2.2 Interaction-based method
The fine-tune procedure is the same to the sentence pair classification task of BERT. The sketch view is shown in Fig. 3. Note that the colon in the figure denotes the concatenation operation. We concatenate the two questions to input it to BERT and use cross entropy loss to train. The full algorithm is shown in Algorithm 2. The fine-tuned model inputs the sentence in the tree node and query sentence as sentence pair to output the score.

![alg1](/assets/png/vector-retrieval/alg1.png)

![alg2](/assets/png/vector-retrieval/alg2.png)


![fig2-3](/assets/png/vector-retrieval/fig23.png)


### 3.3 Tree Building

In this section we describe our tree building strategies. In our tree, each non-leaf node have several child nodes. The leaf nodes contain the real data and the non-leaf nodes are virtual but undertake the function for searching or undertake the function that lead the path to the leaf nodes. 

![table1-2](/assets/png/vector-retrieval/table12.png)

![table3](/assets/png/vector-retrieval/table3.png)


#### 3.3.1 Representation-based method

After all the embeddings of test data are computed, we start to build the tree by k-means. The sketch figure for tree building is shown in Fig. 4. In real the child nodes for each parent may be not that balance. We cluster the embeddings recursively. The sentence embeddings are all in the leaf nodes. The non-leaf node representation is important for the tree search as they pave the way and lead to the right leaf nodes. We use the k-means clustering centers as the non-leaf node embeddings. We think the clustering centers is a good solution for the non-leaf node representation, as it is hard to get the exact representation from the child nodes for the parent nodes. As we already get all the embeddings of test data, we only need to compute the vector distance during tree searching.

#### 3.3.2 Interaction-based method
For interaction-based BERT, we first build the tree by sentence embeddings from the representation-based method above and then use the sentence strings as the leaf nodes. We take the nearest 1-5 sentence strings of cluster centers for the non-leaf node. This strategy has been proved to be effective in experiments.

### 3.4 Tree Search

In this section we describe our tree searching strategies. The two strategies are almost the same. The difference is that representation-based method compute the vector distance at each node but interaction-based method use the deep model to score the string pair at each node.

#### 3.4.1 Representation-based method
At predicting time, we use beam search from top to down to get the nearest top N vectors for the given query vector from the whole tree. If we set the beam size to N, we first choose the top N nodes from the all the child nodes of first level and then search among the chosen child nodes' child nodes for the second level. Then we choose top N nodes from the second level. The detail beam search strategy is shown in Fig 5. 

#### 3.4.2 Interaction-based method
At predicting time, we compute the score of two sentences by BERT for each node while we are searching the tree. As we take 1-5 sentence for a non-leaf node, we use the max similarity score to decide which non-leaf node is better. The detail beam search strategy is the same as Fig 5. shows. The more sentences that are nearest to the clustering centers we take for one non-leaf node, the more computation time we need to do for a non-leaf node. But the most computation time is consumed at the leaf nodes as leaf node number is much larger than non-leaf node number.

