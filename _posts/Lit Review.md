---
layout:     post
title:      "Literature Review"
subtitle:   ""
date:       2020-06-30 02:30:26 +0800
author:     "Ziyu Ye"
header-img: "img/post-bg-apple-event-2015.jpg"
catalog: true
tags:
    - Life
    - Meta
---

### Overview

Anomaly detection in essence can be viewed as a domain adaption task. Our primary goal is to describe a generalization bound for semi-supervised anomaly detection task, using ***task-specific domain descrepancy measure***.

It is non-trivial to find a task-specific domain descrepancy measure. In the following, we review relevant literature topics as following:

- Semi-Supervised Setting in Anomaly Detection
- Learning Theory in Anomaly Detection
- Domain Adaption in Anomaly Detection



### 1. Semi-Supervised Anomaly Detection

#### 1.0. High-Level Idea



### 2. Learning Theory in Anomaly Detection

#### 2.0. High-Level Idea



### 3. Domain Adaption in Anomaly Detection

#### 3.0. High-Level Idea

In domain adaption tasks, there are three major scenarios for data shift, that is:

- **prior shift**, *i.e.* shift of $p(y)$, while $p(x|y)$ is maintained.
- **covariate shift**, *i.e.* shift of $p(x)$, while $p(y|x)$ is maintained.
- **concept shift**, *i.e.* shift of $p(y|x)$, while $p(x)$ is maintained.

The senarios anomaly detection faces may include both **prior shift** and **covariate shift**, in the sense that unknown abnormal data may appear in target domain.

Learning bounds for domain adaption has been rigorously studied. One common idea is to leverage the divergence relation between source and target domains, to upper bound the target risk by the source risk. 

In the following, we will review (1) learning bounds in domain adaption; (2) potential strategy to tackle domain shift; (3) application of transfer learning in domain adaption.



#### 3.1. Divergence Based Learning Bounds

[On the Value of Target Data in Transfer Learning](http://papers.nips.cc/paper/9179-on-the-value-of-target-data-in-transfer-learning)

[Robust Domain Adaption](https://sci-hub.tw/https://doi.org/10.1007/s10472-013-9391-5): Introduces $\lambda$-*shift*, a measure that encapsulates prior knowledge regarding the similarity of source and target domain distributions.



#### 3.2. Potential Strategy to Takle Domain Shift

Sequential adaption



#### 3.3. Application of Transfer Learning in Anomaly Detection

##### 3.3.1. Distribution Shift in Normal Data

[Transfer Anomaly Detection by Inferring Latent Domain Representations (NIPS'20)](https://papers.nips.cc/paper/8517-transfer-anomaly-detection-by-inferring-latent-domain-representations)

[Anomaly Detection with Domain Adaption](https://arxiv.org/abs/2006.03689)







### 4. Miscellaneous

[OOD Detection with Distance Guarantee in Deep Learning](https://arxiv.org/abs/2002.03328)

[Deep Active Learning for Anomaly Detection](https://arxiv.org/pdf/1805.09411.pdf)

[Test-Time Training for OOD Generalization](https://openreview.net/forum?id=HyezmlBKwr)

[Understanding and Improving Information Transfer in Multi-Task Learning (ICLR'20)](https://openreview.net/forum?id=SylzhkBtDB)

[Fixing Bias in Reconstruction-based Anomaly Detection with Lipschitz Discriminators](https://arxiv.org/abs/1905.10710)

[When Does Data Augmentation Help Generalization in NLP](https://arxiv.org/pdf/2004.15012.pdf)

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362v1)

[Learning from Positive and Unlabeled Data: A Survey](https://arxiv.org/abs/1811.04820)