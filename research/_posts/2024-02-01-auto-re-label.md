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
The motivation of randomly re-setting is that the best dataset must be in all randomly possible datasets.

### 1. Introduction

In recent years, deep learning \cite{ref1} model have shown significant improvement on natural language processing(NLP), computer vision and speech processing technologies. However, the model performance is limited by the human labeled data quality. The main reason is that the human labeled data has a certain number of noisy data. Previous work \cite{ref2} has propose the simple idea to find the noisy data and manually correct the noisy data. In this paper, we first review the way we achieve more than 90 score in dev dataset and more than 95 score under human evaluation, then we illustrate our method that randomly re-setting and the best dataset selection.

### 2. Background

![fig1](/assets/png/relabel/fig1.png)

In previous work \cite{ref2}, we illustrate our idea in these steps:

1. It is a classification task. We have a human labeled dataset-v1.

2. We train a deep model upon dataset-v1 and get model-v1.

3. Using model-v1 to predict the classification labels for dataset-v1. 

4. If the predicted labels of dataset-v1 do not equal to the human labels of dataset-v1, we think they are the noisy data.

5. We label the noisy data again by human, while given the labels of model and last label by human as reference. Then we get dataset-v2 and model-v2.

6. We loop these re-label noisy data steps and get the final dataset and final model.

### 3. Auto Re-label

![fig2](/assets/png/auto-relabel/fig1.png)

Our method contains 4 steps:

1. It is a classification task. We have a human labeled dataset-v0 and a model-v0 that is trained on dataset-v0's training dataset.

2. Using model-v1 to predict the classification label for dataset-v0. If the predicted label of dataset-v0 (training dataset and dev dataset) do not equal to the human label of dataset-v0, we think they are the noisy data.

3. We randomly re-set each of the noisy data label to model predicted label or human label. Each setting possibility becomes a new dataset with training + dev dataset. Then we get 2^N datasets (dataset-v1, dataset-v2 ... dataset-vN) and 2^N models (model-v1, model-v2 ... model-vN).

4. We use dev dataset accuracy to select the best dataset from all the 2^N datasets. The best dataset becomes the new dataset-v0.


### 4. Discussion

From the noisy data correction point of view, the noisy data that model prediction and manual label is not equal. 
So any label editing to these noisy data may be the improvement to the dataset quality.

It seems that our method definitely improves the accuracy on the dev dataset, but not necessarily the accuracy of the manual evaluation.
But based on our previous work \cite{ref2}, the accuracy of the dev dataset and the accuracy of the human evaluation are linearly correlated.

We did the manual re-labeling experiments in \cite{ref2}, we verify that the improved dataset exists in all 2^N possible datasets.

### 5. Experiments On Open Datasets

### 6. Conclusion

### References
```
\bibitem{ref1}
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25: 1097-1105.

\bibitem{ref2}
Guo T. The Re-Label Method For Data-Centric Machine Learning[J]. arXiv preprint arXiv:2302.04391, 2023.

```

