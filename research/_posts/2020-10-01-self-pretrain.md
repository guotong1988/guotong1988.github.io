---
layout: post
title: "Self-training For Pre-training Language Models"
date: 2020-10-01
category: research
comments: true
author: "Tong Guo"
description: "A Comprehensive Exploration of Self-training for Pre-training Language Models"
---


# Self-training For Pre-training Language Models

## Abstract

Language model pre-training has proven to be useful in many language understanding tasks. In this paper, we investigate whether it is still helpful to add the self-training method in the pre-training step and the fine-tuning step. Towards this goal, we propose a learning framework that making best use of the unlabel data on the low-resource and high-resource labeled dataset. In industry NLP applications, we have large amounts of data produced by users or customers. Our learning framework is based on this large amounts of unlabel data. First, We use the model fine-tuned on manually labeled dataset to predict pseudo labels for the user-generated unlabeled data. Then we use the pseudo labels to supervise the task-specific training on the large amounts of user-generated data. We consider this task-specific training step on pseudo labels as a pre-training step for the next fine-tuning step. At last, we fine-tune on the manually labeled dataset upon the pre-trained model. In this work, we first empirically show that our method is able to solidly improve the performance by 3.6%, when the manually labeled fine-tuning dataset is relatively small. Then we also show that our method still is able to improve the performance further by 0.2%, when the manually labeled fine-tuning dataset is relatively large enough. We argue that our method make the best use of the unlabel data, which is superior to either pre-training or self-training alone. 

#### keywords
pre-training, self-training, text classification, named entity recognition



## Introduction

![](/assets/png/self-pretrain/fig1.png)


Deep neural networks often require large amounts of labeled data to achieve good performance. However, acquiring labels is a costly process, which motivates research on methods that can effectively utilize unlabeled data to improve performance. Towards this goal, semi-supervised learning \cite{chapelle2009semi} and pre-training are proposed to take advantage of both labeled and unlabeled data. Self-training \cite{scudder1965probability,yarowsky1995unsupervised} is a semi-supervised method which uses a teacher model, trained using labeled data, to create pseudo labels for unlabeled data. Then, a student model is trained with this new training set to yield the final model. Meanwhile, language model pre-training has been shown to be effective for improving many natural language processing (NLP) tasks \cite{devlin2019bert,brown2020language,peters2018deep,radford2019language,radford2018improving}. Previous studies \cite{gururangan2020don,lee2020biobert,beltagy2019scibert} also have shown the benefit of continued pre-training on domain-specific unlabeled data. Then our direct motivation is to answer the question: Do pre-training combined with self-training further improves the fine-tuning performance? Are they complementary? In this paper, we try to answer this question by using the pseudo-label data for pre-training language model.

We are one of the largest Chinese dining and delivery platform. The report shows that the total number of online food orders reaches millions in a day. In our industry NLP applications, we gather large amounts of unlabeled data produced by users or customers. So we have large amounts of data in the food domain, which allows us to use the large amounts of unlabel data. We follow the self-training idea that we can use the fine-tuned model to predict a pseudo-label. We explore the question whether the data with pseudo-label bring further improvement for BERT. We design a learning framework to explore the effect of self-training for pre-training. The whole learning framework is shown in Figure 1. 

In this paper, we evaluate the performance of our learning framework on our item (a specific food) type text classification task and item property named entity recognition (NER) task. For the item type text classification task, our goal is to predict the 1 food type in 311 classes/types for all the 500 million items, given the item name, item short tag written by POI (Point of Interest, a specific store or restaurant) and item's POI name. For the item property NER task, our goal is to extract item property (such as benefit for stomach, sweet and chewy, cool and refreshing) from item description. The task detail is described in section 3.


As it is shown in Figure 1, we have several terminologies to introduce:

#### In-domain data / domain-specific data:
The large amounts of data in our database. Our manually labeled dataset is sampled from this in-domain data. This in-domain data is produced by our user or customer.

#### Self-training:
We use a simple self-training method inspired by Algorithm 1. First, a teacher model is trained on the manually labeled data. Then the teacher model generates pseudo labels on unlabeled data (e.i., all the data in our database). Finally, a student is trained to optimize the loss on human labels and pseudo labels jointly.

#### Pre-training:
This term means the training step that use hundred millions data. The step 1 of Figure 1 is training on the unlabel data. The step 1 of Figure 1 is training on the pseudo-label data. The next step of this training step is the fine-tuning step using the manually labeled data.


#### Self-training as pre-training / task-specific training:
This term refers to the step 4 of Figure 1, which means the training step that use pseudo-label. We consider this training step as a pre-training step for the next fine-tuning step on manually labeled data. 

#### Self-training for fine-tuning / task-specific fine-tuning:
This term refers to the step 5 of Figure 1, which means the fine-tuning step that use both the pseudo-label data and the manually labeled data. 

In detail, we aim to answer the questions: How much does self-training as pre-training perform in low-resource NLP task and high-resource NLP task? To the best of our knowledge, this is the first study exploring the improvement of self-training as pre-training (i.e., using the pseudo-label data for pre-training language models) for natural language understanding. And this is the first study exploring the self-training improvement based on the step of self-training as pre-training (i.e., adding the high-confidence-score pseudo-label data for fine-tuning based on the pre-trained model). This is also the first work exploring the combination of self-training and pre-training when the manually labeled fine-tuning dataset is relatively large (2000K).

In summary, our contributions include:

• We explore the improvement of pre-training combined with self-training. We reveal that pre-training combined with self-training improves the performance stably, when the fine-tuning dataset is relatively small (100K). We find that using the pseudo-label data for fine-tuning do not improve the performance further, when the fine-tuning dataset is relatively large enough. But using the manually labeled data in the fine-tuning step without pseudo-label data improve the performance, when the fine-tuning dataset is relatively large enough (2000K).

• We argue that our learning framework is the best combination of self-training and pre-training method to make use of a large amounts of unlabeled data. Even when the fine-tuning dataset is relatively large enough (2000K), Our method, which is corresponding to Figure 1 and Figure 4-5, still is able to improve the performance. In our experiments based on our dataset, the previous methods (i.e., the classic self-training) is not able to improve the performance when the fine-tuning dataset is relatively large enough (2000K).

• We explore different experimental factors on our text classification dataset and NER dataset. The experiment results prove that the pre-training with task-specific loss and no-MaskLM (masked language model) loss is the best way to make use of the unlabel data. We also find that pre-training using KL-divergence loss with pre-softmax logits is better than cross-entropy loss with one-hot pseudo-label, which is corresponding to the step 4 of Figure 1.


## Related Work

![](/assets/png/self-pretrain/alg1.png)



There is a long history of pre-training language representations\cite{brown1992class,ando2005framework,blitzer2006domain,pennington2014glove,mikolov2013distributed,turian2010word,mnih2009scalable,kiros2015skip,logeswaran2018efficient,jernite2017discourse,hill2016learning}, and we briefly review the most widely-used approaches in this section.

BERT \cite{devlin2019bert} is based on the multi-layer transformer encoder \cite{vaswani2017attention}, which produces contextual token representations that have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task. BERT achieved previously state-of-the-art results on many sentence-level tasks from the GLUE benchmark \cite{wang2018glue}. There are two steps in BERT's framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data by using masked language model task and next sentence prediction task. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. 


Semi-supervised learning \cite{zhu2009introduction,zhu2005semi,chapelle2009semi} solve the problem that making best use of a large amounts of unlabeled data. These works include UDA \cite{xie2020unsupervised}, Mixmatch \cite{berthelot2019mixmatch}, Fixmatch \cite{sohn2020fixmatch}, Remixmatch \cite{berthelot2019remixmatch}. These works design the unsupervised loss to add to the supervised loss. These works prove that domain-specific unlabel data is able to improve the performance, especially in low-resource manually labeled dataset.

Self-training \cite{blum1998combining,zhou2004democratic,zhou2005tri} is one of the earliest and simplest semi-supervised methods. As shown in Algorithm 1, Self-training first uses labeled data to train a good teacher model, then use the teacher model to label unlabeled data and finally use the labeled data and unlabeled data to jointly train a student model. Some early work have successfully applied self-training to word sense disambiguation\cite{yarowsky1995unsupervised} and parsing\cite{huang2009self,reichart2007self,mcclosky2006effective}. In recent years, these works include self-training for natural language processing\cite{du2020self,he2019revisiting}, self-training for computer vision \cite{xie2020self,zoph2020rethinking}, self-training for speech recognition \cite{kahn2020self} and back-translation for machine translation \cite{bojar2011improving,sennrich2015improving,edunov2018understanding}.  \cite{zoph2020rethinking} reveals some generality and flexibility of self-training combined with pre-training in computer vision. \cite{du2020self} study the self-training improvement on the fine-tuning step, based on data augmentation. But \cite{du2020self} restrict to the size of the specific in-domain data and manually labeled data.




## Our Dataset

In this section, we describe our datasets. The item examples are shown in Table 1. The task examples are shown in Table 2. The data size information is shown in Table 3.

![](/assets/png/self-pretrain/table123.png)




### Chinese Text Classification Dataset

This task is to predict the item type or category, given the item name, item short tag, item POI name. We define 311 classes/types for all the items. The model inputs are item name, item short tag given by POI and the POI name of item. We manually labeled 2,040,000 data. The total item number is 500 million.

### Chinese NER Dataset

This task is to extract all the item properties from item description. The item description is a short paragraph written by users. We manually labeled 50,000 data. The total item description number is 200 million. There are 500 million items in total, in which 200 million items have the their descriptions.





## Our Method

In this section, we describe our method in detail. As the Figure 1 shows, our framework includes 5 steps, so we separate this section into 5 subsections: In subsection 4-1, we describe the domain-specific pre-training with unlabeled data. In subsection 4-2, we describe the task-specific fine-tuning with the manually labeled data. In subsection 4-3, we describe the inference step by the fine-tuned model of last step.  In subsection 4-4, we describe the task-specific pre-training with the pseudo-label predicted by the fine-tuned model. In this paper, the step that training the model with the task-specific loss on pseudo-label data is also considered as a pre-training step for the next step. In subsection 4-5, we describe the task-specific fine-tuning which is almost same to subsection 4-2 to get the final model.


### Domain-Specific Pre-training

This is the first step of our method. This step is almost the same to the origin BERT's \cite{devlin2019bert} pre-training step except the data preparation. We use all the in-domain data in our database for pre-training.  

For our Chinese text classification task, the model's 3 inputs are item name, item short tag and item's POI name. We follow the origin BERT \cite{devlin2019bert} setting and use the character-level word masking. In detail we concat the 3 inputs string and mask 15% characters. The max sequence length is 64. We follow the RoBERTa \cite{liu2019roberta} setting and remove the next-sentence-predict loss. For efficiency reason, we use 3-layer-BERT in our experiment because we need to inference more than hundred millions data in our application. We extract 3 layers from the origin official pre-trained BERT \cite{devlin2019bert} as the initialized parameter. The total pre-training data number is 500 million. 

For our Chinese NER task, the model's input is the item description short paragraph. We follow the RoBERTa setting and use the character-level word masking in the short paragraph. The max sequence length is 128 and we remove the next-sentence-predict loss. For efficiency reason, we use 3-layer-BERT in our experiment. We extract 3 layers from the origin official pre-trained BERT as the initialized parameter. The total pre-training data number is 200 million. 


![](/assets/png/self-pretrain/fig2.png)

![](/assets/png/self-pretrain/fig3.png)

![](/assets/png/self-pretrain/fig4.png)

![](/assets/png/self-pretrain/fig5.png)

### Task-Specific Fine-Tuning

This is the second step of our method. We use the model parameter of the last step to initialize this step's model. 

For our Chinese text classification task, we follow the origin BERT \cite{devlin2019bert} setting and use the cross-entropy loss. The total fine-tuning data number is our 2,040,000 manually labeled data. We select 40,000 as test dataset. Then we sampled 100K, 400K and 2000K data from the rest 2000K data separately.

For our Chinese NER task, we use the CRF(Conditional Random Field) loss. The total fine-tuning data number is our 50,000 manually labeled data. We split all the data to 19:1 as training dataset and test dataset.

### Inference Step

This is the third step of our method. This step is to use the fine-tuned model of last subsection to predict a pseudo-label for all the in-domain data.

For our Chinese text classification task, we use the fine-tuned model of the last subsection to predict the classification pseudo-label for all the 500 million unlabeled data. Then we get the 500 million classification data with pseudo-label. The pseudo-label could be the one-hot label or the pre-softmax logits.

For our Chinese NER task, we use the fine-tuned model of the last subsection to predict the sequence tagging pseudo-label for all the 200 million unlabeled data. Then we get the 200 million NER data with pseudo-label. The pseudo-label could be the one-hot label sequence or the pre-softmax logits.

### Task-specific Pre-training

This is the fourth step of our method. We use the pseudo-label of last step as the pseudo ground truth label for training. We consider this training step as the pre-training step for the next step.

For our Chinese text classification task, the model input is the masked text, which is the concat of item name, item short tag and item's POI name. There are four kinds of experiment: 1) The sum of the classification cross-entropy loss and the MaskLM loss. 2) The cross-entropy loss only. 3) The sum of the classification KL-divergence loss on the pre-softmax logits and the MaskLM loss. 4) The KL-divergence loss on the pre-softmax logits only. In detail, we use the [CLS] token's output of BERT for computing the classification cross-entropy loss on one-hot pseudo-label or the KL-divergence loss on the pre-softmax logits. The detail is shown in Figure 2 and Figure 4.

For our Chinese NER task, the model input is masked text, which is the item description short paragraph. There are four kinds of experiment: 1) The sum of the CRF loss and the MaskLM loss. 2) The CRF loss only. 3) The sum of the KL-divergence loss on the pre-softmax logits and the MaskLM loss. 4) The KL-divergence loss on the pre-softmax logits only. In detail, we use all the tokens' output sequence of BERT for computing the CRF loss and MaskLM loss. The detail is shown in Figure 3 and Figure 5.

### Final Task-specific Fine-tuning

This is the final step of our method. We find that fine-tuning on the manually labeled dataset without pseudo-label high-score data is better. For the classic self-training comparison experiment, we randomly sampled the addition data from each class of the pseudo-label high-score data. We use the pre-trained model of subsection 4-4 for fine-tuning. 



## Experiments
In this section we describe our experimental setup and experimental results.

### Experimental Setup

In this section we describe the parameters in our experiments. In all the experiments except the baselines, we use 3-layer-BERT with 768 hidden size and 12 self-attention heads. (Total Parameters = 40M). For efficiency reason, we use only 3 layers of BERT, because it is an industry application and we only have limited computation resource to inference hundred millions data. And 3-layer-BERT is 200%-300% faster than 12-layer-BERT in inference time. The max sequence length for text classification task is 64. The max sequence length for the NER task is 128. The total data size are presented in Table 3. For the text classification task, we setup 3 experimental groups with different fine-tuning data size separately. The 3 different fine-tuning dataset are sampled from the total 2,000,000 data and run the learning framework independently. 

### Text Classification Domain-Specific Pre-training

We use a batch size of 64 * 4-GPU and pre-train for 3,000,000 steps. We use Adam \cite{kingma2014adam} with learning rate of 5e-5, beta1 = 0.9, beta2 = 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout \cite{srivastava2014dropout} probability of 0.1 on all layers. The 3-layer-BERT is initialized by the origin BERT's \cite{devlin2019bert} 3 layers. 

### NER Domain-Specific Pre-training
The NER domain-specific pre-training is almost the same to the text classification domain-specific pre-training. The pre-training data is item description sentence for NER and the pre-training data is concatenation of multi words (item name, item short tag, item poi name) for text classification.

### Text Classification Fine-Tuning
We use a batch size of 64 * 1-GPU and fine-tune for 7 epochs. We use Adam with learning rate of 1e-5. The dropout probability is 0.1 on all layers. The fine-tuning data with different size in Table 2 is sampled from all the 2,000,000 data and the 40,000 test data is fixed.

### NER Fine-Tuning
We use a batch size of 64 * 1-GPU and fine-tune for 3 epochs. We use Adam with learning rate of 1e-5. The dropout probability is 0.1 on all layers.

### Text Classification Task-Specific Pre-training
We use a batch size of 64 * 4-GPU and pre-train for 3,000,000 steps. We use Adam with learning rate of 5e-5, beta1 = 0.9, beta2 = 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. The 3-layer-BERT is initialized by the origin BERT's \cite{devlin2019bert} 3 layers. For fair comparison, we do not initialize this step of pre-training by the result model of domain-specific pre-training step.

### NER Task-Specific Pre-training
We use a batch size of 64 * 4-GPU and pre-train for 3,000,000 steps. We use Adam with learning rate of 5e-5, beta1 = 0.9, beta2 = 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. The 3-layer-BERT is initialized by the origin BERT's \cite{devlin2019bert} 3 layers. For fair comparison, we do not initialize this step of pre-training by the result model of domain-specific pre-training step.



### Baseline Setup
In this section we describe the baselines in Table 2.
### BERT-Base-3layer baseline
 For the BERT-Base-3layer baseline, we extract 3 layers from the origin 12-layer BERT-Base \cite{devlin2019bert} and fine-tune on the manually labeled dataset. 
### BERT-Base-12layer baseline
 For the BERT-Base-12layer baseline, we use the origin 12-layer BERT-Base \cite{devlin2019bert} and fine-tune on the manually labeled dataset. 
### Classic Self-training baseline
 For the Classic Self-training baseline, we use a simple self-training method inspired by Algorithm 1. First, a teacher model is trained on the manually labeled data. Then the teacher model generates pseudo labels on unlabeled data (e.i., all the data in our database). Finally, a student is trained to optimize the loss on human labels and pseudo labels jointly. The pseudo labels data with the high confidence score are averagely sampled from each class of all the data for fine-tuning upon the origin BERT-Base-12layer, Model-A and Model-C. 

## Experimental Results
In this section, we present experiment results on the text classification task and the NER task. The detail results are presented in Table 4-11. The text classification accuracy means the exact match of predicted class and the ground truth. There are 311 classes in the task. The NER F1 is same to the definition of CoNLL \cite{sang2003introduction}.  The BERT-Base baseline's pre-trained model is from the origin official Github repository. The BERT-Base-3layer is extracted from the origin official BERT-Base-12layer.

![](/assets/png/self-pretrain/table4.png)

![](/assets/png/self-pretrain/table5678.png)

![](/assets/png/self-pretrain/table91011.png)

## Analysis

In this section, we analysis our experiment results on Table 4.  we focus on the effect of two factors: the size of the fine-tuning dataset, the data content of the pre-training dataset.

#### Fine-tuning data size
In low-resource dataset, the whole gain of our framework is 3.6% (from 87.0% to 90.6%) for the text classification task. In high-resource dataset, the whole gain of our framework is 1.1% (from 90.8% to 91.9%) for the text classification task. We also observe that pre-training with pseudo-label data (i.e., step 4 of Figure 1) is able to improve the performance in high-resource dataset while fine-tuning with manually labeled data and pseudo-label data is not able to improve the performance in high-resource dataset. In the NER task, the stable improvement of in-domain pre-training (i.e., step 1 of Figure 1) is only 0.2% (from 88.6% to 88.8%), compared to the whole 1.0% (from 88.6% to 89.6%) improvement of the learning framework in the NER task. While the improvement of in-domain pre-training for the text classification task is 0.9% (from 90.8% to 91.9%). And the whole improvement is 1.1% (from 90.8% to 91.9%) for the text classification task.

#### In-domain data and out-domain data
The domain-specific pre-training (i.e., the step 1 of Figure 1) gain of the text classification task is more than the gain of the NER task. The gain of in-domain pre-training (i.e., the step 1 of Figure 1) for text classification task is 0.7% (from 87.5% to 88.2%). The gain of in-domain pre-training (i.e., the step 1 of Figure 1) for the NER task is 0.1% (from 88.7% to 88.8%). The reason is that the input of the text classification task is the concatenation of multi words (item name, item short tag, item poi name), which is not consistent to the origin BERT's \cite{devlin2019bert} pre-training. The origin BERT's pre-training uses the Chinese Wikipedia natural language sentences. And the NER task's in-domain pre-training uses natural language sentences in our database, which is not that much different to the data format of origin BERT.



### Ablation Studies

In this section, we perform ablation experiments over a number of facets in order to better understand their relative importance, which is corresponding to Table 5-8. 

#### In Table 5
we show the ablation experiment results over in-domain pre-training corresponding to step 1 of Figure 1.  The gain of the text classification task (from 87.0% to 88.2%) shows the power of in-domain continued pre-training on unlabel data. Although the gain of NER task (from 88.7% to 88.8%) is small, we speedup the model inference by 200%-300% as we replace the 12-layer BERT to 3-layer BERT.

#### In Table 6
we show the ablation experiment results between unlabel pre-training and pseudo-label pre-training, corresponding to step 1 and 4 of Figure 1. The gain of the text classification task (from 91.7% to 91.9%)  proves that pseudo-label pre-training (i.e., the step 4 of Figure 1) can still improve the performance, even the fine-tuning dataset is very large.

#### In Table 7
we show the ablation experiment results between logits-based loss and one-hot label-based loss. For the task-specific pre-training step on text classification (i.e., the step 4 of Figure 1), using the KL-divergence loss with pre-softmax logits get more gain than the cross-entropy loss with one-hot pseudo-label. We think the reason is that the pre-softmax logits contains more pre-training information than one-hot pseudo-label.

#### In Table 8
we show the ablation experiment results between masked text and the origin text in the task-specific pre-training step. For the task-specific pre-training step (i.e., the step 4 of Figure 1), adding the MaskLM loss or masking 15% tokens of the input text do not get more gain, compared to the no-mask or no-noisy text input. We also observer that no-mask or no-noisy text input for pre-training with pseudo-label is able to improve the performance, even when is fine-tuning dataset is relatively large (2000K).




## Conclusion

In summary, we reveal these conclusions (detail conclusions are shown in Table 9-11):

1) For low-resource manually label dataset, the combination of all the steps of our learning framework improve the performance most, which means pre-training with no-masked logits-based pseudo-label data.

2) For high-resource manually labeled dataset, pre-training with task-specific loss (i.e., step 4 of Figure 1) without masking the input text is the only way to further improve the performance.

3) For pre-training with task-specific loss (i.e., step 4 of Figure 1), input the text without masking is better than masking 15\% tokens of the text.

4) For pre-training with task-specific loss (i.e., step 4 of Figure 1), using the KL-divergence loss computed on pre-softmax logits is better than cross-entropy loss computed on one-hot pseudo-label.

5) More labeled data diminishes the value of self-training. But pre-training still has big gain to performance.

6) Self-training as pre-training (i.e., step 4 of Figure 1) get bigger gain than in-domain pre-training (i.e., step 1 of Figure 1).

In the end, our main contribution is we answered the question: Combining pre-training and self-training to make best use of large amounts of unlabeled data improves the fine-tuning performance. We propose a learning framework using no-masked logits-based pseudo-label data for pre-training, which is superior to either pre-training or self-training alone. The experiment result shows that our learning framework make the best use of the unlabel data even when is fine-tuning dataset is relatively large. 

## Reference

```
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}

@article{brown2020language,
  title={Language models are few-shot learners},
  author={Brown, Tom B and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  journal={arXiv preprint arXiv:2005.14165},
  year={2020}
}

@inproceedings{peters2018deep,
  title={Deep contextualized word representations},
  author={Peters, Matthew E and Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  booktitle={Proceedings of NAACL-HLT},
  pages={2227--2237},
  year={2018}
}

@article{gururangan2020don,
  title={Don't Stop Pretraining: Adapt Language Models to Domains and Tasks},
  author={Gururangan, Suchin and Marasovi{\'c}, Ana and Swayamdipta, Swabha and Lo, Kyle and Beltagy, Iz and Downey, Doug and Smith, Noah A},
  journal={arXiv preprint arXiv:2004.10964},
  year={2020}
}

@article{liu2019roberta,
  title={Roberta: A robustly optimized bert pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}

@inproceedings{wang2018glue,
  title={GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
  booktitle={Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP},
  pages={353--355},
  year={2018}
}

@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}

@article{lee2020biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining.},
  author={Lee, J and Yoon, W and Kim, S and Kim, D and Kim, S and So, CH and Kang, J},
  journal={Bioinformatics (Oxford, England)},
  volume={36},
  number={4},
  pages={1234},
  year={2020}
}

@inproceedings{sang2003introduction,
  title={Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition},
  author={Sang, Erik Tjong Kim and De Meulder, Fien},
  booktitle={Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003},
  pages={142--147},
  year={2003}
}

@inproceedings{beltagy2019scibert,
  title={SciBERT: A Pretrained Language Model for Scientific Text},
  author={Beltagy, Iz and Lo, Kyle and Cohan, Arman},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3606--3611},
  year={2019}
}

@inproceedings{pennington2014glove,
  title={Glove: Global vectors for word representation},
  author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
  booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)},
  pages={1532--1543},
  year={2014}
}

@article{zoph2020rethinking,
  title={Rethinking pre-training and self-training},
  author={Zoph, Barret and Ghiasi, Golnaz and Lin, Tsung-Yi and Cui, Yin and Liu, Hanxiao and Cubuk, Ekin D and Le, Quoc V},
  journal={arXiv preprint arXiv:2006.06882},
  year={2020}
}

@article{du2020self,
  title={Self-training Improves Pre-training for Natural Language Understanding},
  author={Du, Jingfei and Grave, Edouard and Gunel, Beliz and Chaudhary, Vishrav and Celebi, Onur and Auli, Michael and Stoyanov, Ves and Conneau, Alexis},
  journal={arXiv preprint arXiv:2010.02194},
  year={2020}
}

@article{srivastava2014dropout,
  title={Dropout: a simple way to prevent neural networks from overfitting},
  author={Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan},
  journal={The journal of machine learning research},
  volume={15},
  number={1},
  pages={1929--1958},
  year={2014},
  publisher={JMLR. org}
}

@article{xie2020unsupervised,
  title={Unsupervised data augmentation for consistency training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Thang and Le, Quoc},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{mcclosky2006effective,
  title={Effective self-training for parsing},
  author={McClosky, David and Charniak, Eugene and Johnson, Mark},
  booktitle={Proceedings of the Human Language Technology Conference of the NAACL, Main Conference},
  pages={152--159},
  year={2006}
}

@inproceedings{reichart2007self,
  title={Self-training for enhancement and domain adaptation of statistical parsers trained on small datasets},
  author={Reichart, Roi and Rappoport, Ari},
  booktitle={Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics},
  pages={616--623},
  year={2007}
}

@inproceedings{huang2009self,
  title={Self-training PCFG grammars with latent annotations across languages},
  author={Huang, Zhongqiang and Harper, Mary},
  booktitle={Proceedings of the 2009 conference on empirical methods in natural language processing},
  pages={832--841},
  year={2009}
}

@inproceedings{yarowsky1995unsupervised,
  title={Unsupervised word sense disambiguation rivaling supervised methods},
  author={Yarowsky, David},
  booktitle={33rd annual meeting of the association for computational linguistics},
  pages={189--196},
  year={1995}
}

@article{he2019revisiting,
  title={Revisiting self-training for neural sequence generation},
  author={He, Junxian and Gu, Jiatao and Shen, Jiajun and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1909.13788},
  year={2019}
}

@inproceedings{xie2020self,
  title={Self-training with noisy student improves imagenet classification},
  author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10687--10698},
  year={2020}
}

@inproceedings{bojar2011improving,
  title={Improving translation model by monolingual data},
  author={Bojar, Ond{\v{r}}ej and Tamchyna, Ale{\v{s}}},
  booktitle={Proceedings of the Sixth Workshop on Statistical Machine Translation},
  pages={330--336},
  year={2011}
}

@article{sennrich2015improving,
  title={Improving neural machine translation models with monolingual data},
  author={Sennrich, Rico and Haddow, Barry and Birch, Alexandra},
  journal={arXiv preprint arXiv:1511.06709},
  year={2015}
}

@article{edunov2018understanding,
  title={Understanding back-translation at scale},
  author={Edunov, Sergey and Ott, Myle and Auli, Michael and Grangier, David},
  journal={arXiv preprint arXiv:1808.09381},
  year={2018}
}

@inproceedings{berthelot2019mixmatch,
  title={Mixmatch: A holistic approach to semi-supervised learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin A},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5049--5059},
  year={2019}
}

@article{sohn2020fixmatch,
  title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
  author={Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Zhang, Han and Raffel, Colin},
  journal={arXiv preprint arXiv:2001.07685},
  year={2020}
}

@article{berthelot2019remixmatch,
  title={Remixmatch: Semi-supervised learning with distribution alignment and augmentation anchoring},
  author={Berthelot, David and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Sohn, Kihyuk and Zhang, Han and Raffel, Colin},
  journal={arXiv preprint arXiv:1911.09785},
  year={2019}
}

@inproceedings{kahn2020self,
  title={Self-training for end-to-end speech recognition},
  author={Kahn, Jacob and Lee, Ann and Hannun, Awni},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7084--7088},
  year={2020},
  organization={IEEE}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}

@article{radford2018improving,
  title={Improving language understanding by generative pre-training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  year={2018}
}

@article{brown1992class,
  title={Class-based n-gram models of natural language},
  author={Brown, Peter F and Della Pietra, Vincent J and Desouza, Peter V and Lai, Jennifer C and Mercer, Robert L},
  journal={Computational linguistics},
  volume={18},
  number={4},
  pages={467--480},
  year={1992}
}


@article{ando2005framework,
  title={A framework for learning predictive structures from multiple tasks and unlabeled data.},
  author={Ando, Rie Kubota and Zhang, Tong and Bartlett, Peter},
  journal={Journal of Machine Learning Research},
  volume={6},
  number={11},
  year={2005}
}

@inproceedings{blitzer2006domain,
  title={Domain adaptation with structural correspondence learning},
  author={Blitzer, John and McDonald, Ryan and Pereira, Fernando},
  booktitle={Proceedings of the 2006 conference on empirical methods in natural language processing},
  pages={120--128},
  year={2006}
}

@article{mikolov2013distributed,
  title={Distributed representations of words and phrases and their compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  journal={arXiv preprint arXiv:1310.4546},
  year={2013}
}

@inproceedings{turian2010word,
  title={Word representations: a simple and general method for semi-supervised learning},
  author={Turian, Joseph and Ratinov, Lev and Bengio, Yoshua},
  booktitle={Proceedings of the 48th annual meeting of the association for computational linguistics},
  pages={384--394},
  year={2010}
}

@inproceedings{mnih2009scalable,
  title={A scalable hierarchical distributed language model},
  author={Mnih, Andriy and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1081--1088},
  year={2009},
  organization={Citeseer}
}

@article{kiros2015skip,
  title={Skip-thought vectors},
  author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
  journal={arXiv preprint arXiv:1506.06726},
  year={2015}
}

@article{logeswaran2018efficient,
  title={An efficient framework for learning sentence representations},
  author={Logeswaran, Lajanugen and Lee, Honglak},
  journal={arXiv preprint arXiv:1803.02893},
  year={2018}
}

@article{jernite2017discourse,
  title={Discourse-based objectives for fast unsupervised sentence representation learning},
  author={Jernite, Yacine and Bowman, Samuel R and Sontag, David},
  journal={arXiv preprint arXiv:1705.00557},
  year={2017}
}

@article{hill2016learning,
  title={Learning distributed representations of sentences from unlabelled data},
  author={Hill, Felix and Cho, Kyunghyun and Korhonen, Anna},
  journal={arXiv preprint arXiv:1602.03483},
  year={2016}
}

@article{zhu2009introduction,
  title={Introduction to semi-supervised learning},
  author={Zhu, Xiaojin and Goldberg, Andrew B},
  journal={Synthesis lectures on artificial intelligence and machine learning},
  volume={3},
  number={1},
  pages={1--130},
  year={2009},
  publisher={Morgan \& Claypool Publishers}
}

@article{zhu2005semi,
  title={Semi-supervised learning literature survey},
  author={Zhu, Xiaojin Jerry},
  year={2005},
  publisher={University of Wisconsin-Madison Department of Computer Sciences}
}

@article{chapelle2009semi,
  title={Semi-supervised learning (chapelle, o. et al., eds.; 2006)[book reviews]},
  author={Chapelle, Olivier and Scholkopf, Bernhard and Zien, Alexander},
  journal={IEEE Transactions on Neural Networks},
  volume={20},
  number={3},
  pages={542--542},
  year={2009},
  publisher={IEEE}
}

@inproceedings{blum1998combining,
  title={Combining labeled and unlabeled data with co-training},
  author={Blum, Avrim and Mitchell, Tom},
  booktitle={Proceedings of the eleventh annual conference on Computational learning theory},
  pages={92--100},
  year={1998}
}

@inproceedings{zhou2004democratic,
  title={Democratic co-learning},
  author={Zhou, Yan and Goldman, Sally},
  booktitle={16th IEEE International Conference on Tools with Artificial Intelligence},
  pages={594--602},
  year={2004},
  organization={IEEE}
}

@article{zhou2005tri,
  title={Tri-training: Exploiting unlabeled data using three classifiers},
  author={Zhou, Zhi-Hua and Li, Ming},
  journal={IEEE Transactions on knowledge and Data Engineering},
  volume={17},
  number={11},
  pages={1529--1541},
  year={2005},
  publisher={IEEE}
}

@article{scudder1965probability,
  title={Probability of error of some adaptive pattern-recognition machines},
  author={Scudder, H},
  journal={IEEE Transactions on Information Theory},
  volume={11},
  number={3},
  pages={363--371},
  year={1965},
  publisher={IEEE}
}
```
