---
layout: post
title: "Improving Text Generation for Product Description via Human Behaviour"
date: 2024-02-02
category: research
author: "Tong Guo"
description: "Improving Text Generation for Product Description via Human Behaviour"
---
# Improving Text Generation for Product Description via Human Behaviour

### Abstract
Text generation is an important method to generate high quality and available product description from product title. For the product description generation for online E-commerce application, the main problem is how to improve the quality of generated text. In other words, how we judge the quality of text. If all texts are already positive and available, then we find it impossible to manually judge which text is the better text for a product. So if we cannot judge which is a better text manually,  we cannot improve the quality of generated text. In E-commerce, product description is to attract shoppers and improve sales. So we design a method to improve the quality of generated text based on user buying behaviour. Online result shows that our approach improve the sales of products by improving the text quality.

### Introduction

In recent years, the development of deep learning (DL) has brought breakthroughs on text generation. The sequence to sequence (Seq2Seq) models use encoder-decoder transformer \cite{vaswani2017attention} for better model flexibility. The most representative models of this
type include T5 \cite{raffel2020exploring} and BART \cite{lewis2020bart}. In this paper, we adopt the T5 \cite{raffel2020exploring} model to conduct our data-centric experiments. 

In E-commerce, product description can attract shoppers and improve sales. But manually writing a successful product description is highly time-consuming. Text generation \cite{zhang2022survey,prabhumoye2020exploring} technologies play a crucial role in this range of applications.


Text generation has an input or a source sequence $X$ and an output or a target sequence $Y$ to be generated. In our product description generation tasks, $X$ is the product title and $Y$ is the product description. The examples are shown in Table \ref{table1}.


The problem is to improve the quality of the generated text in E-commerce, then the following problem is how we define and judge what is high quality text. We find it impossible to manually judge which text is the better text for a product. To answer this problem, we define text that brings more sales to a product as better text. So we need to use user buying behaviour to judge which are better texts.


Then the problem is how to use user buying behaviour to select better quality text, the problem is that the text is displayed alongside the product, and we need to isolate the impact of the gain of the text on the product.

In summary, in order to improve the quality of generated description text for product. We face these problems: 

1) We cannot judge which text is the better text for a product by human labeling. If we can't judge and define text as good or bad, then we can't optimize it.

2) We use user buying behaviour to judge better text. Then we need to isolate the gain impact of the text. Because text is displayed on the product. The product descriptions and products are bound together to be displayed to the users.

3) We need to design a complete solution to solve the above problems all together.

In this paper, we solve these problems and propose these contributions:

1) We use user buying behaviour to judge which text is better for a product, in order to solve the problem that we cannot judge it manually.

2) We train a sales prediction model upon the user buying logs of our E-commerce application. In order to isolate the gain impact of the text for the product, we use causal inference method.

3) We design a complete solution to continuously improve the quality of generated product description, guided by user behaviour.

### Method

The whole pipeline is shown in Figure \ref{fig1}. In this paper, we adopt the T5 \cite{raffel2020exploring} model to conduct our text generation experiments. We adopt transformer \cite{vaswani2017attention} as our sales prediction model.

Our method contains 8 steps: 

In Step-1, we get an initial dataset to train the T5 generative model. The initial dataset is constructed by query ChatGPT. We ask ChatGPT to write product description, input the product title as the prompts. We then remove the data in the training dataset that do not suitable to display online.

In Step-2, we use the T5 model to generate product descriptions for hundreds of millions the products. 

In Step-3, we display the generated product descriptions on the products. 

In Step-4, we collect the logs of sales and views of each product.

In Step-5, we train the sales prediction model based on the online logs. The detail is illustrated in the following sections.

In Step-6 and Step-7, we use causal inference to find out the best quality product descriptions in the logs. The detail is illustrated in the following sections.

In Step-8, we retrain the T5 model using the quality text identified of the last step. Then we do AB experiments to evaluate the performance of the generated product description of online App.

#### Initial Training Dataset Construction

This section corresponding to the Step-1 in Figure \ref{fig1}.
We collect our initial training dataset by querying ChatGPT. Each prompt is formed by concatenating a product title. We ask ChatGPT to write descriptions for the products. We tried to add some product attributes as prompt, but most of the ChatGPT's results do not relate to the product attributes. Table \ref{table1} shows the prompt examples and the ChatGPT's results. Our T5 \cite{raffel2020exploring} model trained on this initial dataset gets 88\% available rate, under human evaluation. 

#### Sales Prediction Model

After the generated product description has been displayed on the online products, The users view and buy the products. So now the problem is that we want to train a sales prediction model for products. The training target is:

$$ RPM = N_{sales} / N_{views} $$

where $N_{sales}$ is the sales amount of product and $N_{views}$ is the viewed amount by users to this product.

The input features for the model include:

1) Product related features: product title, product tags, product history RPM.

2) Product description: Text tokens.

#### Training Target

We designed the training objective to capture the additional gain of text for product. So our training objective is the absolute value of the RPM, and we use the regression loss.

$$ Loss = |RPM - model\_output| $$


#### Causal Inference

The role of causal inference in our approach is to be used to isolate the impact of text for product after having a trained sales prediction model. 
When we make a prediction, we input the product features and the text to get score $A$, and only the product features to get score $B$. The gain effect of the text for the product is $A-B$. The detail framework of sales prediction and causal inference is shown in Figure \ref{fig2}.

### Experiment

#### Manual Evaluation

The manual evaluation contains two parts: the generation available rate, the comparison of the two generation results. The available rate is to determine whether the generated text is available. We use human annotation to compute:

$$ Rate = N_{good} / N_{total} $$

where $N_{good}$ is the available generated text number and $N_{total}$ is the total texts that are human annotated. 

The manual comparison of the Step-1 initial results and the optimised model results. The initial model results are corresponding to the Step-1 of the Figure \ref{fig1}. The optimised model results are the generated texts from the optimised model of the Step-8 of the Figure \ref{fig1}. We find it impossible to manually determine which of the two is more appropriate to be displayed on a product.

The available rate result is shown in Table \ref{table3}. The comparison of the two generation result is shown is shown in Table \ref{table4}.

#### AB Experiments

We do AB experiments to evaluate the online performance of our method. We display the two product descriptions to two groups of users. Then we count the RPM of the two groups of users. The results show that the optimised texts improve the RPM by about 0.1\%, compared to another group.



### Discussion

In this section we illustrate the reason why we design our method. And we illustrate the baseline solution we compare.

#### Motivation

In order to improve the quality of the generated text, and to improve the RPM, we found that manual annotation can not achieve this, we look for supervisory signals from human behaviour of online App.

#### Baseline Solution

We first collect the results from ChatGPT to train our T5 model. We input product title and ask ChatGPT to write product description. The problems with the ChatGPT results are that the available rate of text is 89\% and 11\% of the product description is not suitable for display. So we clean the dataset based on ChatGPT API and train the T5 model with more than 99\% available rate of generated text. We use this generated results of our T5 model as the baseline, which is the Step-1 of Figure \ref{fig1}.



### Relate Work

#### Text Generation
The pre-trained model based on Transformer \cite{vaswani2017attention} has greatly improved the performance in text generation. The learning objectives include masked language modeling (MLM) and causal language modeling (CLM). MLM-based Language Models include BERT \cite{devlin2018bert}, ROBERTA \cite{liu2019roberta}. CLM-based Language Models include the GPT series works \cite{radford2018improving,radford2019language,brown2020language} and other decoder-only transformer models \cite{keskar2019ctrl}.
The sequence to sequence models\cite{sutskever2014sequence} use encoder-decoder transformer \cite{vaswani2017attention} for better model flexibility. The seq2seq model is widely used in the field of text generation \cite{luong2014addressing,bahdanau2014neural}. We adopt Seq2Seq models implement our text generation tasks. The most representative models of this type include T5 \cite{raffel2020exploring} and BART \cite{lewis2020bart}. In this paper, we adopt the T5 model to conduct our experiments. We compared T5 and GPT-2\cite{radford2019language} on the same dataset and ultimately chose T5.

#### CTR Prediction
Sales prediction task \cite{tsoumakas2019survey,cheriyan2018intelligent} is to estimate future sales of products, which is same to the click through rate (CTR) prediction task \cite{chen2016deep,guo2017deepfm}. Sales prediction and CTR prediction both use users behaviour (click/view) as training target, which means that we collect the logs of online App to build the training dataset.

#### Causal Inference
The research questions that motivate most quantitative studies in the health, social and behavioral sciences are not statistical but causal in nature. Causal inference \cite{pearl2010causal,pearl2009causal} is to solve these problems. In our scenario, the product description is displayed on the product. That is, the product description, is the treatment that affects the sales of the corresponding product.

#### Text Quality Evaluation
The evaluation of text generation \cite{celikyilmaz2020evaluation,zhang2019bertscore} is the task that evaluate of natural language generation, for example in machine translation and caption
generation, requires comparing candidate sentences to annotated references. In our scenario, however, we are unable to manually evaluate the impact of the quality of the generated product description on product sales. So we do AB experiments to count whether the generated text leads to an increase in product sales or not, to judge whether the quality of text is improved.



### Conclusion

How to improve the quality of generated text is a very critical issue, as manual annotation cannot judge the quality of generated text. If manual annotation cannot judge the quality of the generated text, then we cannot optimise the text generation to a better quality direction. On the other hand, if we can find a method to judge the quality of the generated text, then we can continuously optimise it. This paper we find supervised signals in the E-commerce scenario that can continuously optimise the quality of generated text. We have developed a complete solution and sales of our App have been boosted.

### References
```
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Ben-
gio. 2014. Neural machine translation by jointly
learning to align and translate. arXiv preprint
arXiv:1409.0473.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners. Advances in neural information processing
systems, 33:1877–1901.

Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao.
2020. Evaluation of text generation: A survey. arXiv
preprint arXiv:2006.14799.

Junxuan Chen, Baigui Sun, Hao Li, Hongtao Lu, and
Xian-Sheng Hua. 2016. Deep ctr prediction in dis-
play advertising. In Proceedings of the 24th ACM
international conference on Multimedia, pages 811–
820.

Sunitha Cheriyan, Shaniba Ibrahim, Saju Mohanan, and
Susan Treesa. 2018. Intelligent sales prediction using
machine learning techniques. In 2018 International
Conference on Computing, Electronics & Communi-
cations Engineering (iCCECE), pages 53–58. IEEE.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2018. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. arXiv preprint arXiv:1810.04805.

Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li,
and Xiuqiang He. 2017. Deepfm: a factorization-
machine based neural network for ctr prediction.
arXiv preprint arXiv:1703.04247.

Nitish Shirish Keskar, Bryan McCann, Lav R Varshney,
Caiming Xiong, and Richard Socher. 2019. Ctrl: A
conditional transformer language model for control-
lable generation. arXiv preprint arXiv:1909.05858.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
Veselin Stoyanov, and Luke Zettlemoyer. 2020. Bart:
Denoising sequence-to-sequence pre-training for nat-
ural language generation, translation, and comprehen-
sion. In Proceedings of the 58th Annual Meeting of
the Association for Computational Linguistics, pages
7871–7880.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach. arXiv preprint arXiv:1907.11692.

Minh-Thang Luong, Ilya Sutskever, Quoc V Le, Oriol
Vinyals, and Wojciech Zaremba. 2014. Addressing
the rare word problem in neural machine translation.
arXiv preprint arXiv:1410.8206.

Judea Pearl. 2009. Causal inference in statistics: An
overview.

Judea Pearl. 2010. Causal inference. Causality: objec-
tives and assessment, pages 39–58.

Shrimai Prabhumoye, Alan W Black, and Rus-
lan Salakhutdinov. 2020. Exploring control-
lable text generation techniques. arXiv preprint
arXiv:2005.01822.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya
Sutskever, et al. 2018. Improving language under-
standing by generative pre-training.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, et al. 2019. Language
models are unsupervised multitask learners. OpenAI
blog, 1(8):9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J Liu. 2020. Exploring the limits
of transfer learning with a unified text-to-text trans-
former. The Journal of Machine Learning Research,
21(1):5485–5551.

Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014.
Sequence to sequence learning with neural networks.
Advances in neural information processing systems,
27.

Grigorios Tsoumakas. 2019. A survey of machine learn-
ing techniques for food sales prediction. Artificial
Intelligence Review, 52(1):441–447.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. Advances in neural information processing
systems, 30.

Hanqing Zhang, Haolin Song, Shaoyu Li, Ming Zhou,
and Dawei Song. 2022. A survey of controllable
text generation using transformer-based pre-trained
language models. arXiv preprint arXiv:2201.05337.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-
uating text generation with bert. arXiv preprint
arXiv:1904.09675.
```

