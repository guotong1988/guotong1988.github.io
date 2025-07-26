---
layout: post
title: "Self-Predict And Manual-Select For Improving LoRA-based Domain Fine-tuning"
date: 2025-07-01
category: research
author: "Tong Guo"
description: ""
---
# Self-Predict And Manual-Select For Improving LoRA-based Domain Fine-tuning
### Abstract

LoRA fine-tuning preserves the information of the base LLMs
while incorporating domain-specific data through fine-tuning. 
Therefore, if we use QA-pair domain training dataset to LoRA fine-tune a LLM
and then employ this fine-tuned LLM to predict the domain training dataset itself, 
we can prepare two or more answers for each QA-pair's question. 
We manually label the optimal answer from the answers, replace the original answer, 
and proceed to the next round of LoRA fine-tuning.
Thus, we can continuously optimize the training dataset through iterative self-predict and human-select.
This method can also be applied to multi-turn QA fine-tuning datasets.
The human evaluation results of the fine-tuned LLM demonstrate that our approach is effective.


### 1. Introduction

In recent years, the development of large language models (LLMs) \cite{ref1,ref2,ref3} has brought breakthroughs on NLP applications. 

Parameter Efficient Fine-Tuning (PEFT) \cite{ref4,ref5} provides a practical solution by efficiently adjusting the large models over the various downstream tasks.

The proposed method can be iteratively applied to continuously enhance the quality of the dataset.

### 2. Method

![fig1](/assets/png/self-select/fig1.png)

### 3. Experiments

### 4. Discussion

If we intend to fine-tune a LLM for domain-specific applications, then acquiring the step-1 dataset, as the foundation of the entire workflow, carries greater weight compared with the method proposed in this paper.‌ For instance, when employing API-based prompt engineering (e.g., utilizing GPT-4 or Gemini) for dataset generation, ‌rigorous optimization of the prompt engineering pipeline itself becomes paramount‌ to secure a high-quality seed training dataset.




### 5. Conclusion

### Reference
```

\bibitem{ref1}
Achiam J, Adler S, Agarwal S, et al. Gpt-4 technical report[J]. arXiv preprint arXiv:2303.08774, 2023.

\bibitem{ref2}
Yang A, Li A, Yang B, et al. Qwen3 technical report[J]. arXiv preprint arXiv:2505.09388, 2025.

\bibitem{ref3}
Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models[J]. arXiv preprint arXiv:2302.13971, 2023.

\bibitem{ref4}
Hu E J, Shen Y, Wallis P, et al. Lora: Low-rank adaptation of large language models[J]. ICLR, 2022, 1(2): 3.

\bibitem{ref5}
Han Z, Gao C, Liu J, et al. Parameter-efficient fine-tuning for large models: A comprehensive survey[J]. arXiv preprint arXiv:2403.14608, 2024.
```
