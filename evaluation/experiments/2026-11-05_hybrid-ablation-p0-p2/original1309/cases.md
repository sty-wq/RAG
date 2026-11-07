# 代表性错误与互补样本

样本按排名退化幅度选取，并补充 BM25 带来收益的题。判断“正确块”沿用金标准段落匹配，不代表人工回答判定。

## 1907.12984-q01

In the paper "DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting", which datasets do they evaluate on?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 1 | 0.5000 |
| bm25 | 0 | 0.0000 |
| rrf_a0 | 1 | 0.5000 |
| rrf_a0.25 | 0 | 0.0000 |
| rrf_a0.5 | 0 | 0.0000 |
| rrf_a1 | 0 | 0.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| e8d3c7ec4035ff591d12b65e242b7db1 | 1 | 1 | 1 | 1 | [] |
| a70a9b6789dab7ad675be41579185b9f | 2 | 3 | 2 | 2 | [] |
| 71ff8a1572380b069dbbeaa9543894fe | 3 | 31 | 11 | 5 | [] |
| c66baeeea0784de7f59b214761778dc6 | 4 | 未进池 | 40 | 20 | [1, 2] |
| 59ac1bff152685f63f01b06252405bc8 | 5 | 39 | 17 | 6 | [] |
| cd0d3772bf5e4c539e6b14869b3e01e3 | 8 | 2 | 3 | 4 | [] |
| 302afa3ef3e31c56376ecb8d428db491 | 6 | 6 | 4 | 3 | [] |
| a9594bb4945c7c61c65d12ec0076be8e | 16 | 7 | 5 | 11 | [] |

### e8d3c7ec4035ff591d12b65e242b7db1

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Evaluation

We conduct multiple experiments to evaluate the effectiveness of our system in many ways.


### a70a9b6789dab7ad675be41579185b9f

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Introduction

The contributions of this paper can be concluded into the following aspects:


### 71ff8a1572380b069dbbeaa9543894fe

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Data Description

We firstly run the standard Transformer model on the NIST dataset. Then we evaluate the quality of the pre-trained model on our proposed speech translation dataset, and propose effective methods to improve the performance of the baseline. In that the testing data in this dataset contains ASR errors and speech irregularities, it can be used to evaluate the robustness of novel methods.

In the final deployment, we train our model using a corpus containing approximately 200 million bilingual pairs both in Chinese-English and English-Chinese translation tasks.


### c66baeeea0784de7f59b214761778dc6

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Data Description

We use the streaming multi-layer truncated attention model (SMLTA) trained on the large-scale speech corpus (more than 10,000 hours) and fine-tuned on a number of talk related corpora (more than 1,000 hours), to generate the 5-best automatic recognized text for each acoustic speech.

The test dataset includes interpretations produced by simultaneous interpreters with professional experience. This dataset contributes an essential resource for the comparison between translation and interpretation.

We randomly extract several talks from the dataset, and divide them into the development and test set. In Table TABREF34 , we summarize the statistics of our dataset. The average number of utterances per talk is 152.6 in the training set, 59.75 in the dev set, and 162.5 in the test set.


### 59ac1bff152685f63f01b06252405bc8

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Experiments

As shown in Table TABREF74 , it is clear that DuTongChuan achieves promising acceptability on both translation tasks (85.71% for Chinese-English, and 86.36 % for English-Chinese). We also elaborately analyze the error types in the final translations, and we find that apart from errors occurring in translation and ASR, a majority of errors come from IU boundary detection, which account for nearly a half of errors. In the future, we should concentrate on improving the translation quality by enhancing the robustness of our IU boundary detector. We also evaluate the latency of our model in an End-to-End manner (speech-to-speech), and we find that the target speech slightly lags behind the source speech in less than 3 seconds at most times.


### cd0d3772bf5e4c539e6b14869b3e01e3

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Conclusion and Future Work

In this paper, we propose DuTongChuan, a novel context-aware translation model for simultaneous interpreting. This model is able to constantly read streaming text from the ASR model, and simultaneously determine the boundaries of information units one after another. The detected IU is then translated into a fluent translation with two simple yet effective decoding strategies: partial decoding and context-aware decoding. We also release a novel speech translation corpus, BSTC, to boost the research on robust speech translation task.


### 302afa3ef3e31c56376ecb8d428db491

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Experiments

The overall performance both on translation quality and latency reveals that DuTongChuan is accessible and practicable in an industrial scenario.


### a9594bb4945c7c61c65d12ec0076be8e

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Abstract

In this paper, we present DuTongChuan, a novel context-aware translation model for simultaneous interpreting. This model allows to constantly read streaming text from the Automatic Speech Recognition (ASR) model and simultaneously determine the boundaries of Information Units (IUs) one after another. The detected IU is then translated into a fluent translation with two simple yet effective decoding strategies: partial decoding and context-aware decoding. In practice, by controlling the granularity of IUs and the size of the context, we can get a good trade-off between latency and translation quality easily.

### 金标准证据

证据 0

Recently, we release Baidu Speech Translation Corpus (BSTC) for open research . This dataset covers speeches in a wide range of domains, including IT, economy, culture, biology, arts, etc. We transcribe the talks carefully, and have professional translators to produce the English translations. This procedure is extremely difficult due to the large number of domain-specific terminologies, speech redundancies and speakers' accents. We expect that this dataset will help the researchers to develop robust NMT models on the speech translation. In summary, there are many features that distinguish this dataset to the previously related resources:

证据 1

The test dataset includes interpretations produced by simultaneous interpreters with professional experience. This dataset contributes an essential resource for the comparison between translation and interpretation.

证据 2

We randomly extract several talks from the dataset, and divide them into the development and test set. In Table TABREF34 , we summarize the statistics of our dataset. The average number of utterances per talk is 152.6 in the training set, 59.75 in the dev set, and 162.5 in the test set.

证据 3

We use a subset of the data available for NIST OpenMT08 task . The parallel training corpus contains approximate 2 million sentence pairs. We choose NIST 2006 (NIST06) dataset as our development set, and the NIST 2002 (NIST02), 2003 (NIST03), 2004 (NIST04) 2005 (NIST05), and 2008 (NIST08) datasets as our test sets. We will use this dataset to evaluate the performance of our partial decoding and context-aware decoding strategy from the perspective of translation quality and latency.

## 1909.13466-q01

In the paper "Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation", what baselines do they compare to?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 1 | 1.0000 |
| bm25 | 0 | 0.0000 |
| rrf_a0 | 1 | 1.0000 |
| rrf_a0.25 | 0 | 0.0000 |
| rrf_a0.5 | 0 | 0.0000 |
| rrf_a1 | 0 | 0.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| f7373b0237f9d8f9603c1ddccbef41b2 | 1 | 未进池 | 40 | 13 | [1] |
| 707399f0a0c9ee4b5a9a880e1462a8e9 | 2 | 11 | 2 | 1 | [] |
| 52577c5e9317542c606bb6871853547a | 3 | 未进池 | 41 | 15 | [] |
| 2ecaf3a9efc60a981928c76718cd164c | 4 | 33 | 9 | 4 | [] |
| fb7d8bf307d44560335c6b4d9476f823 | 5 | 36 | 15 | 5 | [] |
| b9566544bccc830ca40d15c45ae28610 | 9 | 1 | 1 | 3 | [] |
| 15c54b6c007b044c4746f5fbcb505a71 | 14 | 3 | 3 | 8 | [] |
| 337b80e80e39944753e88789490ea01d | 6 | 12 | 4 | 2 | [] |
| 60588b073486748fcd80a935e8a230a1 | 12 | 8 | 5 | 7 | [] |

### f7373b0237f9d8f9603c1ddccbef41b2

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > The Baseline NMT model

In this section, we describe the NMT model that has been used as the basis for the proposed regularizer. It is a neural encoder-decoder architecture with attention BIBREF1 that can be regarded as a strong baseline as it incorporates both LSTMs and transformers as modules. Let us assume that $\textbf {x}:\lbrace x_1 \dots x_n\rbrace $ is the source sentence with $n$ tokens and $\textbf {y}:\lbrace y_1 \dots y_m\rbrace $ is the target translated sentence with $m$ tokens. First, the words in the source sentence are encoded into their word embeddings by an embedding layer:

and then the source sentence is encoded by a sequential module into its hidden vectors, ${\textbf {h}_1 \dots \textbf {h}_n}$:


### 707399f0a0c9ee4b5a9a880e1462a8e9

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > Introduction

The rest of this paper is organized as follows. Section SECREF2 presents and discusses the related work. Section SECREF3 describes the model used as baseline while Section SECREF4 presents the proposed regularization techniques, ReWE and ReSE. Section SECREF5 describes the experiments and analyzes the experimental results. Finally, Section SECREF6 concludes the paper.


### 52577c5e9317542c606bb6871853547a

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > The Baseline NMT model

Given the context vector, $\textbf {c}_j$, the decoder output at the previous step, $\textbf {s}_{j-1}$, and the word embedding of the previous word in the target sentence, $\textbf {y}^{e}_{j}$ (Eq. DISPLAY_FORM18), the decoder generates vector $\textbf {s}_j$ (Eq. DISPLAY_FORM19). This vector is later transformed into a larger vector of the same size as the target vocabulary via learned parameters $\textbf {W}$, $\textbf {b}$ and a softmax layer (Eq. DISPLAY_FORM20). The resulting vector, $\textbf {p}_j$, is the inferred probability distribution over the target vocabulary at decoding step $j$. Fig. FIGREF12 depicts the full architecture of the baseline model.

The model is trained by minimizing the negative log-likelihood (NLL) which can be expressed as:


### 2ecaf3a9efc60a981928c76718cd164c

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > The Baseline NMT model

where the probability of ground-truth word ${y}_j$ has been noted as $\textbf {p}_{j}({y}_{j})$. Minimizing the NLL is equivalent to MLE and results in assigning maximum probability to the words in the reference translation, $y_j, j=1 \ldots m$. The training objective is minimized with standard backpropagation over the training data, and at inference time the model uses beam search for decoding.


### fb7d8bf307d44560335c6b4d9476f823

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > Experiments > Results

We have carried out a number of experiments with both baselines. The scores reported are an average of the BLEU scores (in percentage points, or pp) BIBREF46 over the test sets of 5 independently trained models. Table TABREF44 shows the results over the en-fr dataset. In this case, the models with ReWE have outperformed the LSTM and transformer baselines consistently. The LSTM did not benefit from using BPE, but the transformer+ReWE with BPE reached $36.30$ BLEU pp (a $+0.99$ pp improvement over the best model without ReWE). For this dataset we did not use ReSE because French was the target language.


### b9566544bccc830ca40d15c45ae28610

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > Regressing word and sentence embeddings

As mentioned in the introduction, MLE suffers from some limitations when training a neural machine translation system. To alleviate these shortcomings, in our recent paper BIBREF8 we have proposed a new regularization method based on regressing word embeddings. In this paper, we extend this idea to sentence embeddings.


### 15c54b6c007b044c4746f5fbcb505a71

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > Related Work

The related work is organized over the three main research subareas that have motivated this work: regularization techniques, word and sentence embeddings and unsupervised NMT.


### 337b80e80e39944753e88789490ea01d

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > Conclusion

Finally, we have showed that the regularized models have also outperformed the baselines in experiments on unsupervised NMT. As future work, we plan to explore how the categorical and continuous predictions from our model could be jointly utilized to further improve the quality of the translations.


### 60588b073486748fcd80a935e8a230a1

1909.13466.md > Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation > Related Work > Regularization Techniques

and Zhang et al.BIBREF23 have introduced two regularization terms based on the Kullback-Leibler (KL) divergence to improve the agreement of sentences predicted from left-to-right and right-to-left.

### 金标准证据

证据 0

Extensive experimentation over four language pairs of different dataset sizes (from small to large) with both word and sentence regularization. We show that using both ReWE and ReSE can outperform strong state-of-the-art baselines based on long short-term memory networks (LSTMs) and transformers.

证据 1

In this section, we describe the NMT model that has been used as the basis for the proposed regularizer. It is a neural encoder-decoder architecture with attention BIBREF1 that can be regarded as a strong baseline as it incorporates both LSTMs and transformers as modules. Let us assume that $\textbf {x}:\lbrace x_1 \dots x_n\rbrace $ is the source sentence with $n$ tokens and $\textbf {y}:\lbrace y_1 \dots y_m\rbrace $ is the target translated sentence with $m$ tokens. First, the words in the source sentence are encoded into their word embeddings by an embedding layer:

## 1711.04964-q01

In the paper "Dynamic Fusion Networks for Machine Reading Comprehension", how much improvement is given on RACE by their introduced approach?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 1 | 1.0000 |
| bm25 | 0 | 0.0000 |
| rrf_a0 | 1 | 1.0000 |
| rrf_a0.25 | 0 | 0.0000 |
| rrf_a0.5 | 0 | 0.0000 |
| rrf_a1 | 0 | 0.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| 5708290d00e68a6420e13f68f4862d59 | 1 | 2 | 1 | 1 | [] |
| 16d83dc2f6e039d009770f51da3bbc29 | 2 | 7 | 4 | 2 | [] |
| de9ec0bbf33ae74652f24060cc210ddd | 3 | 3 | 3 | 3 | [] |
| 95765ade86f893b5f072f16b5dc8939c | 4 | 50 | 22 | 8 | [0] |
| d1bac0bea1a21c3a8ba9550c9b2df7e9 | 5 | 1 | 2 | 4 | [] |
| 532e70101f826e8b1553cbef52b19a51 | 7 | 6 | 5 | 5 | [] |

### 5708290d00e68a6420e13f68f4862d59

1711.04964.md > Dynamic Fusion Networks for Machine Reading Comprehension > Related Work

we will further illustrate this in Section " RACE - The MRC Task" .


### 16d83dc2f6e039d009770f51da3bbc29

1711.04964.md > Dynamic Fusion Networks for Machine Reading Comprehension > Introduction

Experiments conducted on the RACE dataset show that DFN significantly outperforms previous state-of-the-art MRC models and has achieved the best result reported on RACE. A thorough empirical analysis also demonstrates that DFN is highly effective in understanding passages of a wide variety of styles and answering questions of different complexities.


### de9ec0bbf33ae74652f24060cc210ddd

1711.04964.md > Dynamic Fusion Networks for Machine Reading Comprehension > RACE - The MRC Task

In this section, we first give a brief introduction to the RACE dataset, and then explain the rationale behind choosing RACE as the testbed in our study.


### 95765ade86f893b5f072f16b5dc8939c

1711.04964.md > Dynamic Fusion Networks for Machine Reading Comprehension > Model Performance

Table 3 shows a comparison between DFN and a few previously proposed models. All models were trained with the full RACE dataset, and tested on RACE-M and RACE-H, respectively. As shown in the table, on RACE-M, DFN leads to a 7.8% and 7.3% performance boost over GA and Stanford AR, respectively. On RACE-H, the outperformance is 1.5% and 2.7%. The ensemble models also gained a performance boost of 4-5% comparing to previous methods. We suspect that the lower gain on RACE-H might result from the higher level of difficulty in those questions in RACE-H, as well as ambiguity in the dataset. Human performance drops from 85.1 on RACE-M to 69.4 on RACE-H, which indicates RACE-H is very challenging even for human.


### d1bac0bea1a21c3a8ba9550c9b2df7e9

1711.04964.md > Dynamic Fusion Networks for Machine Reading Comprehension > Introduction

The proposed model is called a Dynamic Fusion Network (DFN). In this paper, we describe the version of DFN developed on the RACE dataset BIBREF7 . In RACE, a list of candidate answers is provided for each passage-question pair. So DFN for RACE is a scoring model - the answer candidate with the highest score will be selected as the final answer.


### 532e70101f826e8b1553cbef52b19a51

1711.04964.md > Dynamic Fusion Networks for Machine Reading Comprehension > Experiments

To evaluate the proposed DFN model, we conducted experiments on the RACE dataset. Statistics of the training/dev/test data are provided in Table 2 . In this section, we present the experimental results, with a detailed analysis on the dynamic selection of strategies and multi-step reasoning. An ablation study is also provided to demonstrate the effectiveness of dynamic fusion and reasoning in DFN.

### 金标准证据

证据 0

Table 3 shows a comparison between DFN and a few previously proposed models. All models were trained with the full RACE dataset, and tested on RACE-M and RACE-H, respectively. As shown in the table, on RACE-M, DFN leads to a 7.8% and 7.3% performance boost over GA and Stanford AR, respectively. On RACE-H, the outperformance is 1.5% and 2.7%. The ensemble models also gained a performance boost of 4-5% comparing to previous methods. We suspect that the lower gain on RACE-H might result from the higher level of difficulty in those questions in RACE-H, as well as ambiguity in the dataset. Human performance drops from 85.1 on RACE-M to 69.4 on RACE-H, which indicates RACE-H is very challenging even for human.

证据 1

To avoid variance in training and provide a fair comparison, 3 ensembles of each model were trained and evaluated on both dev and test sets. As shown in Table 5 , the DFN model has a 1.6% performance gain over the basic model (without DF and MR). This performance boost was contributed by both multi-step reasoning and dynamic fusion. When omitting DF or MR alone, the performance of DFN model dropped by 1.1% and 1.2%, respectively.

## 1708.05482-q01

In the paper "A Question Answering Approach to Emotion Cause Extraction", what was their system's f1 score?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 1 | 1.0000 |
| bm25 | 0 | 0.0000 |
| rrf_a0 | 1 | 1.0000 |
| rrf_a0.25 | 0 | 0.0000 |
| rrf_a0.5 | 0 | 0.0000 |
| rrf_a1 | 0 | 0.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| 7267a48afa42f7811474fde5cb90efc9 | 1 | 6 | 1 | 1 | [] |
| 828ba099f1b138f1bd59c63a63f73b04 | 2 | 8 | 3 | 2 | [] |
| 0fb9e15ebdb30f101f0fd1c25de46b64 | 3 | 37 | 12 | 5 | [] |
| 7acb0ca6defbe5c9c9a21bb8b89dca53 | 4 | 45 | 20 | 7 | [0] |
| 84456452fb296e110f3bbb65a289f1a0 | 5 | 4 | 2 | 3 | [] |
| 65fa82cca6bcdfb4131e55d2a4aceaf6 | 10 | 1 | 4 | 6 | [] |
| ddc7e611e97a149071dd4b7efae776e4 | 9 | 3 | 5 | 4 | [] |

### 7267a48afa42f7811474fde5cb90efc9

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Abstract

Performance evaluation shows that our method achieves the state-of-the-art performance on a recently released emotion cause dataset, outperforming a number of competitive baselines by at least 3.01% in F-measure.


### 828ba099f1b138f1bd59c63a63f73b04

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Experiments and Evaluation

We first presents the experimental settings and then report the results in this section.


### 0fb9e15ebdb30f101f0fd1c25de46b64

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Experimental Setup and Dataset

In the experiments, we randomly select 90% of the dataset as training data and 10% as testing data. In order to obtain statistically credible results, we evaluate our method and baseline methods 25 times with different train/test splits.


### 7acb0ca6defbe5c9c9a21bb8b89dca53

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Evaluation and Comparison

Table 2 shows the evaluation results. The rule based RB gives fairly high precision but with low recall. CB, the common-sense based method, achieves the highest recall. Yet, its precision is the worst. RB+CB, the combination of RB and CB gives higher the F-measure But, the improvement of 1.27% is only marginal compared to RB.


### 84456452fb296e110f3bbb65a289f1a0

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Conclusions

In the future, we will explore effective ways [id=lq]to model discourse relations among clauses and develop a QA system which can directly output the cause of emotions as answers.


### 65fa82cca6bcdfb4131e55d2a4aceaf6

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Introduction

Existing approaches to emotion cause extraction mostly rely on methods typically used in information extraction, such as rule based template matching, sequence labeling and classification based methods. Most of them use linguistic rules or lexicon features, but do not consider the semantic information and ignore the relation between the emotion word and emotion cause. In this paper, we present a new method for emotion cause extraction. We consider emotion cause extraction as a question answering (QA) task. Given a text containing the description of an event which [id=lq]may or may not cause a certain emotion, we take [id=lq]an emotion word [id=lq]in context, such as “sad”, as a query. The question to the QA system is: “Does the described event cause the emotion of sadness?”.


### ddc7e611e97a149071dd4b7efae776e4

1708.05482.md > A Question Answering Approach to Emotion Cause Extraction > Introduction

The rest of the paper is organized as follows. Section SECREF2 gives a review of related works on emotion analysis. Section SECREF3 presents our proposed deep memory network based model for emotion cause extraction. Section SECREF4 discusses evaluation results. Finally, Section SECREF5 concludes the work and outlines the future directions.

### 金标准证据

证据 0

Table 2 shows the evaluation results. The rule based RB gives fairly high precision but with low recall. CB, the common-sense based method, achieves the highest recall. Yet, its precision is the worst. RB+CB, the combination of RB and CB gives higher the F-measure But, the improvement of 1.27% is only marginal compared to RB.

## 1611.06322-q06

In the paper "Spotting Rumors via Novelty Detection", how do they define rumors?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 0 | 0.0000 |
| bm25 | 1 | 1.0000 |
| rrf_a0 | 0 | 0.0000 |
| rrf_a0.25 | 1 | 1.0000 |
| rrf_a0.5 | 1 | 1.0000 |
| rrf_a1 | 1 | 1.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| 7ccd511d73c89d8ae31d0a9f2961bc5e | 1 | 28 | 10 | 4 | [] |
| ec12cf0ec8da5bc64d132b11b7cdab66 | 2 | 14 | 5 | 2 | [] |
| 54e905a183a380e80e8b07627d7427ee | 3 | 3 | 1 | 1 | [] |
| 45220cbef5fd3ec4fda6e92e5ee17452 | 4 | 8 | 2 | 3 | [] |
| 71ec47cad08ec5bf0cc4fe35db6d2e60 | 5 | 17 | 8 | 6 | [] |
| 39c2dec31d5fd55d928a9b42cf15b30e | 7 | 5 | 3 | 5 | [0] |
| f2b91bc388b74900d330d828645d4717 | 9 | 6 | 4 | 7 | [] |

### 7ccd511d73c89d8ae31d0a9f2961bc5e

1611.06322.md > Spotting Rumors via Novelty Detection > Abstract

Rumour detection is hard because the most accurate systems operate retrospectively, only recognising rumours once they have collected repeated signals. By then the rumours might have already spread and caused harm. We introduce a new category of features based on novelty, tailored to detect rumours early on. To compensate for the absence of repeated signals, we make use of news wire as an additional data source. Unconfirmed (novel) information with respect to the news articles is considered as an indication of rumours. Additionally we introduce pseudo feedback, which assumes that documents that are similar to previous rumours, are more likely to also be a rumour.


### ec12cf0ec8da5bc64d132b11b7cdab66

1611.06322.md > Spotting Rumors via Novelty Detection > Introduction

In a nutshell, documents that are similar to previously detected rumours are considered to be more likely to also be a rumour. The proposed features can be computed in constant time and space allowing us to process high-volume streams in real-time (Muthukrishnan, 2005). Our experiments reveal that novelty based features and pseudo feedback significantly increases detection performance for early rumour detection.


### 54e905a183a380e80e8b07627d7427ee

1611.06322.md > Spotting Rumors via Novelty Detection > Related Work

In contrast, we introduce new features to detect rumours as early as possible - preferably instantly, allowing them to be debunked before they spread and cause harm.


### 45220cbef5fd3ec4fda6e92e5ee17452

1611.06322.md > Spotting Rumors via Novelty Detection > Introduction

The contributions of this paper include:

Novelty based Features

We introduced a new category of features for instant rumour detection that harnesses trusted resources. Unconfirmed (novel) information with respect to trusted resources is considered as an indication of rumours.

Pseudo Feedback for Detection/Classification

Pseudo feedback increases detection accuracy by harnessing repeated signals, without the need of retrospective operation.


### 71ec47cad08ec5bf0cc4fe35db6d2e60

1611.06322.md > Spotting Rumors via Novelty Detection > Detecting unpopular rumours

Previous approaches to rumour detection rely on repeated signals to form propagation graphs or clustering methods. Beside causing a detection delay these methods are also blind to less popular rumours that don't go viral. In contrast, novelty based feature require only a single message enabling them to detect even the smallest rumours. Examples for such small rumours are shown in table 3 .


### 39c2dec31d5fd55d928a9b42cf15b30e

1611.06322.md > Spotting Rumors via Novelty Detection > Rumour Detection

We introduce two new classes of features, one based on novelty, the other on pseudo feedback. Both feature categories improve detection accuracy early on, when information is limited.


### f2b91bc388b74900d330d828645d4717

1611.06322.md > Spotting Rumors via Novelty Detection > Abstract

Comparison with other real-time approaches shows that novelty based features in conjunction with pseudo feedback perform significantly better, when detecting rumours instantly after their publication.

### 金标准证据

证据 0

Rumour detection is a challenging task, as it requires determining the truth of information (Zhao et. al, 2015). The Cambridge dictionary, defines a rumour as information of doubtful or unconfirmed truth. We rely on classification using an SVM, which is the state-of-the-art approach for novelty detection. Numerous features have been proposed for rumour detection on social media, many of which originate from an original study on information credibility by Castillo et. al (2011). Unfortunately, the currently most successful features rely on information based on graph propagation and clustering, which can only be computed retrospectively. This renders them close to useless when detecting rumours early on. We introduce two new classes of features, one based on novelty, the other on pseudo feedback. Both feature categories improve detection accuracy early on, when information is limited.

证据 1

Social Media has evolved from friendship based networks to become a major source for the consumption of news (NIST, 2008). On social media, news is decentralised as it provides everyone the means to efficiently report and spread information. In contrast to traditional news wire, information on social media is spread without intensive investigation, fact and background checking. The combination of ease and fast pace of sharing information provides a fertile breeding ground for rumours, false- and disinformation. Social media users tend to share controversial information in-order to verify it, while asking about for the opinions of their followers (Zhao et. al, 2015). This further amplifies the pace of a rumour's spread and reach. Rumours and deliberate disinformation have already caused panic and influenced public opinion.

证据 2

To increase instantaneous detection performance, we compensate for the absence of future information by consulting additional data sources. In particular, we make use of news wire articles, which are considered to be of high credibility. This is reasonable as according to Petrovic et. al (2013), in the majority of cases, news wires lead social media for reporting news. When a message arrives from a social media stream, we build features based on its novelty with respect to the confirmed information in the trusted sources. In a nutshell, the presence of information unconfirmed by the official media is construed as an indication of being a rumour. Note that this closely resembles the definition of what a rumour is.

## 1812.00382-q01

In the paper "Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions", which model architecture do they opt for?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 0 | 0.0000 |
| bm25 | 1 | 1.0000 |
| rrf_a0 | 0 | 0.0000 |
| rrf_a0.25 | 0 | 0.0000 |
| rrf_a0.5 | 0 | 0.0000 |
| rrf_a1 | 1 | 1.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| 11db1857873bd219497a051b7824d77b | 1 | 7 | 2 | 2 | [] |
| 3b06e03eb1a28effb06a34837248a923 | 2 | 2 | 1 | 1 | [] |
| 36011480e7b748632f584ed7a1e72d79 | 3 | 15 | 7 | 5 | [] |
| 5b661451e8a011634bfa9752d20f1fcd | 4 | 8 | 6 | 4 | [] |
| 35c643ed86022833e3b2c6581175a190 | 5 | 3 | 3 | 3 | [] |
| e1bd691f1868c5a19d17834414b59eea | 6 | 4 | 4 | 6 | [] |
| 2caa096fc2a4bf4d3434b3ef9e2baad0 | 11 | 1 | 5 | 9 | [1] |

### 11db1857873bd219497a051b7824d77b

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Conclusion

Controversy detection is a hard task, as it forms a latent concept sensitive to vocabulary gaps between topics and vocabulary shifts over time. We analysed the performance of language model, SVM, CNN and HAN models on different tasks.


### 3b06e03eb1a28effb06a34837248a923

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Introduction & Prior work

We take the task of controversy detection and evaluate robustness of different methodologies with respect to the varying nature of controversies.


### 36011480e7b748632f584ed7a1e72d79

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Introduction & Prior work

The downside of these approaches is the lack of generalizability due to their platform-specific nature. 3) matching models that combine lexical and explicit modelling approaches by looking at lexical similarities between a given text and a set of texts in a domain that provides explicit features BIBREF1 , BIBREF6 , BIBREF7 .


### 5b661451e8a011634bfa9752d20f1fcd

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Abstract

By leveraging the semantic properties of word embeddings we robustly improve on existing controversy detection methods. To evaluate model stability over time and to unseen topics, we asses model performance under varying training conditions to test cross-temporal, cross-topic, cross-domain performance and annotator congruence. In doing so, we demonstrate that weak-signal based neural approaches are closer to human estimates of controversy and are more robust to the inherent variability of controversies.


### 35c643ed86022833e3b2c6581175a190

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Robustness of the model across topics

These results indicate that neural methods seem better able to generalize to unseen topics.


### e1bd691f1868c5a19d17834414b59eea

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Abstract

Detecting controversy in general web pages is a daunting task, but increasingly essential to efficiently moderate discussions and effectively filter problematic content. Unfortunately, controversies occur across many topics and domains, with great changes over time. This paper investigates neural classifiers as a more robust methodology for controversy detection in general web pages. Current models have often cast controversy detection on general web pages as Wikipedia linking, or exact lexical matching tasks. The diverse and changing nature of controversies suggest that semantic approaches are better able to detect controversy. We train neural networks that can capture semantic information from texts using weak signal data.


### 2caa096fc2a4bf4d3434b3ef9e2baad0

1812.00382.md > Improved and Robust Controversy Detection in General Web Pages Using Semantic Approaches under Large Scale Conditions > Methods

The optimization algorithm used is Adam BIBREF13 (learning rate: 1e-3).

### 金标准证据

证据 0

The majority of NLP-task related neural architectures rely on word embeddings, popularized by Mikolov et al BIBREF9 to represent texts. In essence these embeddings are latent-vector representations that aim to capture the underlying meaning of words. Distances between such latent-vectors are taken to express semantic relatedness, despite having different surface forms. By using embeddings, neural architectures are also able to leverage features learned on other texts (e.g. pretrained word embeddings) and create higher level representations of input (e.g. convolutional feature maps or hidden-states). These properties suggest that neural approaches are better able to generalize to unseen examples that poorly match the training set. We use two often applied network architectures adopting word embeddings, to classify controversy: Recurrent Neural Networks BIBREF10 and Convolutional Neural Networks BIBREF11 to answer the following research question. RQ: Can we increase robustness of controversy detection using neural methods?

证据 1

We explore the potential of RNNs and CNNs for controversy detection using both the HAN BIBREF10 and the CNN BIBREF11 model. Similar to BIBREF10 , each bi-directional GRU cell is set to a dimension of 50, resulting in a word/sentence representation of size 100 after concatenation. The word/sentence attention vectors similarly contain 100 dimensions, all randomly initialized. The word windows defined in the CNN model are set to sizes: 2, 3 and 4 with 128 feature maps each. Each model is trained using mini batches of size 64 and uses both dropout (0.5) and INLINEFORM0 regularization (1e-3) at the dense prediction layer. Both networks use pre-trained embeddings, trained on 100 billion words of a Google News corpus, which are further fine-tuned during training on the controversy dataset. The optimization algorithm used is Adam BIBREF13 (learning rate: 1e-3).

