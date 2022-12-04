---
layout: post
title: "Self-Refine Learning For Data-Centric Text Classification"
date: 2021-09-01
category: research
comments: true
author: "Tong Guo"
description: "Self-Refine Learning For Data-Centric Text Classification"
---



# Self-Refine Learning For Data-Centric Text Classification

### Abstract

In industry NLP application, our manually labeled data has a certain number of noisy data. We present a simple method to find the noisy data and re-label their labels to the result of model prediction. We select the noisy data whose human label is not contained in the top-K model's predictions. The model is trained on the origin dataset. The experiment result shows that our method works. For industry deep learning application, our method improve the text classification accuracy from 80.5% to 90.6% in dev dataset, and improve the human-evaluation accuracy from 83.2% to 90.1%.


#### Keywords
Deep Learning, Text Classification, NLP

### 1. Introduction

In recent years, deep learning \cite{ref2} and BERT-based \cite{ref1} model have shown significant improvement on almost all the NLP tasks. However, the most important factor for deep learning application performance is the data quantity and quality. We try to improve performance of the industry NLP application by correcting the noisy data by other most of data. 

Previous works \cite{ref11} first find the noisy data which human label and model prediction is not equal and re-label the noisy data manually. During the correction, the last human label and model prediction is viewed by the labeling people. However it need more human labeling. So in this work, we directly re-label the noisy data whose label is not contained in the top-K (K=1,2,3...10) predictions of model. We re-label the noisy data's label to the top-1 prediction of model. 

Our key contribution is:

Based on our industry dataset, we first find the noisy data which human label is not in the top-K (K=1,2,3...10)  predictions of model. Then we re-label the label of noisy data to the top-1 prediction of the model. The experiment results shows that our idea works for our large industry dataset.  

![](/assets/png/self-refine/fig1.png)

 

### 2. Relate Works

BERT \cite{ref1} is built by the multi-layer transformer encoder \cite{ref10}, which produces self-attended token representations that have been pre-trained from unlabeled text and fine-tuned for the supervised downstream tasks. BERT achieved state-of-the-art results on many sentence-level tasks on the GLUE benchmark \cite{ref3} and CLUE\cite{ref12} benchmark. 


Our method is different to semi-supervised learning. Semi-supervised learning solve the problem that making best use of a large amount of unlabeled data. These works include UDA \cite{ref6}, Mixmatch \cite{ref7}, Fixmatch \cite{ref8}, Remixmatch \cite{ref9}. Our work is full supervised.

### 3. Our Method

In this section, we describe our method in detail. Our methods is shown in Fig 1. It includes 5 steps:

Step 1, in order to solve our industry text classification problem. We manually label 2,790,000 data and split them into 2,700,000 training data and 90,000 dev data. 

Step 2, we train / fine-tune the BERT model on the 2,700,000 training data. We named the result model of this step Model-A. Note that Model-A should not overfit the training dataset.

Step 3, we use Model-A to predict for all the 2,790,000 data. Then we find all the data whose human label are not in the top-K (K=1,2,3...10) predictions of model-A. We consider they are the noisy data. 

Step 4, we re-label the noisy data's human label to the top-1 prediction of model-A. Then we split the same 2,700,000:90,000 training and dev dataset.

Step 5, we train and evaluate upon the dataset of step 4 and get Model-B. As we also re-label the dev dataset by the top-1 prediction of model-A, we also manually evaluate the performance of our method.


### 4. The Model

We use BERT as our model. The training steps in our method belongs to the fine-tuning step in BERT. We follow the BERT convention to encode the input text. 



### 5. Experiments

In this section we describe detail of experiment parameters and show the experiment result. The detail result is shown in Table 2. The data size in our experiment is shown in Table 1.

In fine-tuning, we use Adam \cite{ref4} with learning rate of 1e-5 and use a dropout \cite{ref5} probability of 0.1 on all layers. We use BERT-Base (12 layer, 768 hidden size) as our pre-trained model. 

![](/assets/png/self-refine/table1.png)



![](/assets/png/self-refine/table2.png)




### 6. Analysis

As the whole dataset is large enough, so we re-label the ground truth of noisy data by other most of the data.

To illustrate why self-refine work. We illustrate that only dropping the noise data work.

Take text classification as example. If there are 2-class to classify. The sample training data is like:

——————

‘aac’ – class-A

‘aad’ – class-A

‘aae’ – class-B – wrong-label

‘bbc’ – class-B

‘bbd’ – class-B

‘bbe’ – class-B

——————

We define that the text started with ‘aa’ to label class-A and started with ‘bb’ to label class-B.

Then the trained model will inference ‘aae’ to class-A, but will inference ‘aaee’ to class-B, according to the training data distribution.

Then we drop the wrong-label training data and get:

——————

‘aac’ – class-A

‘aad’ – class-A

‘bbc’ – class-B

‘bbd’ – class-B

‘bbe’ – class-B

——————

Then the trained model will inference ‘aae’ to class-A, and also will inference ‘aaee’ to class-A, according to the training data distribution.

The reason for this example is: A little wrong-label data can lead to wrong inference for some kind of new data, but the wrong-label data can be found by self-predict-and-compare method.

### 7. Conclusion

The experiment result shows our idea works. Our idea can apply to a broad set of deep learning industry applications. We will do the experiments like \cite{ref11} that inject the prediction result of model-A to model-B. For further applying of self-correct method, we can correct the noise data which model prediction and human label is not equal, while the model prediction confidence score is high. But we still encourage the human re-label method of \cite{ref13}.


### Reference
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
Guo, Tong. "Learning From How Human Correct." arXiv preprint arXiv:2102.00225 (2021).

\bibitem{ref12}
Xu, Liang, et al. "Clue: A chinese language understanding evaluation benchmark." arXiv preprint arXiv:2004.05986 (2020).

\bibitem{ref13}
Tong Guo, Re-Label Is All You Need, Guotong1988. Github. Io.
```

