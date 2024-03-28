---
layout: post
title: "Automatic Label Error Correction"
date: 2024-02-01
category: core_research
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

In recent years, deep learning \cite{ref1} model have shown significant improvement on natural language processing(NLP), computer vision and speech processing technologies. However, the model performance is limited by the human labeled data quality. The main reason is that the human labeled data has a certain number of noisy data. Previous work \cite{ref2} has propose the simple idea to find the noisy data and manually correct the noisy data. In this paper, we first review the way we achieve more than 90 score in dev dataset and more than 95 score under human evaluation, then we illustrate our method that randomly re-setting and the best dataset selection. The code is at [github.com/guotong1988/Automatic-Label-Error-Correction](https://github.com/guotong1988/Automatic-Label-Error-Correction)

### 2. Background

![fig1](/assets/png/auto-relabel/fig1.png)

In previous work \cite{ref2}, we illustrate our idea in these steps:

Step-1. It is a classification task. We have a human labeled dataset-v0. We train a deep model upon dataset-v0 and get model-v0.

Step-2. Using model-v0 to predict the classification labels for dataset-v0. If the predicted labels of dataset-v0 do not equal to the human labels of dataset-v0, we think they are the noisy data.

Step-3. We label the noisy data again by human, while given the labels of model and last label by human as reference. Then we get dataset-v1

Step-4. We train on dataset-v1 and get model-v1.

We loop these re-labeling noisy data steps and get the final dataset and final model.

### 3. Auto Re-label

![fig2](/assets/png/auto-relabel/fig2.png)

Our method contains 4 steps:

Step-1. It is a classification task. We have a human labeled dataset-v0 and a model-v0 that is trained on dataset-v0's training dataset.

Step-2. Using model-v1 to predict the classification label for dataset-v0. If the predicted label of dataset-v0 (training dataset and dev dataset) do not equal to the human label of dataset-v0, we think they are the noisy data.

Step-3. We randomly re-set each of the noisy data label to model predicted label or human label. Each setting possibility becomes a new dataset with training + dev dataset. Then we get 2^N datasets (dataset-v1, dataset-v2 ... dataset-vN) and 2^N models (model-v1, model-v2 ... model-vN).

Step-4. We use dev dataset accuracy to select the best dataset from all the 2^N datasets. The best dataset becomes the new dataset-v0.


### 4. Discussion

From the noisy data correction point of view, the noisy data that model prediction and manual label is not equal. 
So any label editing to these noisy data may be the improvement to the dataset quality.

It seems that our method definitely improves the accuracy on the dev dataset, but not necessarily the accuracy of the manual evaluation.
But based on our previous work \cite{ref2}, the accuracy of the dev dataset and the accuracy of the human evaluation are linearly correlated.

We did the manual re-labeling experiments in \cite{ref2}, we verify that the improved dataset exists in all 2^N possible datasets.

### 5. Experiments On Open Datasets

The experiment result on TREC-6 text classification dataset is shown in Table \cite{table1}. The model is pretrained BERT \cite{ref3}, which has 12-layer and 768 hidden size.

![table1](/assets/png/auto-relabel/table1.png)

### 6. Related Work 

Data-centric \cite{ref5} approach which focuses on label quality by characterizing and identifying label errors in datasets.

\cite{ref7} proposes pseudo-label-based method to improve data quality without re-label, which is different from our method. Our method is to improve the data quality for model of 97\% accuracy/precision/recall/BLEU/AUC by re-label.

\cite{ref8} proposes label-guess-based method to improve data quality without re-label, which is different from our method. Our method get the guess-label as reference for re-label.

\cite{ref9} use model predictions for human as references to label. \cite{ref9} use the new human-labeled data from model predictions to train the reward model, which is named as reinforcement learning from human feedback(RLHF).

Our work do not have the reward model of RLHF, the differnce between RLHF and our re-label method is that RLHF focus on using the model predictions as training dataset of reward model, and our re-label method focus on correcting the origin training dataset of policy model. Also in RLHF, the new human-labeled data from model predictions do not conflict to our method, because the new data can be simply merged to the whole dataset of policy model, and then to re-label/re-correct by our method.

For object detection task, \cite{ref11} uses the re-label method and surpasses human performance in offline LiDAR based 3D object detection.

### 7. Conclusion

In previous works, we manually re-label the noisy data to improve the dataset quality. 
In this paper, we propose the method to automatically re-label the noisy data to improve the dataset quality. The experimental results and theoretical analysis verify our idea: The best re-labeled dataset exists in the all possible candidate datasets.

### References
```
\bibitem{ref1}
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25: 1097-1105.

\bibitem{ref2}
Guo T. The Re-Label Method For Data-Centric Machine Learning[J]. arXiv preprint arXiv:2302.04391, 2023.

\bibitem{ref3}
Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

\bibitem{ref4}
Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.

\bibitem{ref5}
Zha D, Bhat Z P, Lai K H, et al. Data-centric artificial intelligence: A survey[J]. arXiv preprint arXiv:2303.10158, 2023.

\bibitem{ref7}
Yun S, Oh S J, Heo B, et al. Re-labeling imagenet: from single to multi-labels, from global to localized labels[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 2340-2350.

\bibitem{ref8}
Nicholson B, Zhang J, Sheng V S, et al. Label noise correction methods[C]//2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015: 1-9.

\bibitem{ref9}
Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. arXiv preprint arXiv:2203.02155, 2022.

\bibitem{ref10}
Guo, Tong (2022): Re-Label By Data Pattern For Knowledge Driven Deep Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.20485917.v6

\bibitem{ref11}
Fan L, Yang Y, Mao Y, et al. Once Detected, Never Lost: Surpassing Human Performance in Offline LiDAR based 3D Object Detection[J]. arXiv preprint arXiv:2304.12315, 2023.

```

