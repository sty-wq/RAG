# 代表性错误与互补样本

样本按排名退化幅度选取，并补充 BM25 带来收益的题。判断“正确块”沿用金标准段落匹配，不代表人工回答判定。

## 1911.10401-q01

Which baseline models are compared against a transformer-based recurrent convolutional neural network for irony and sarcasm detection?

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
| db6bd302e70ad317ba170b061636fc0f | 1 | 1 | 1 | 1 | [] |
| a710fac01ff2981442bfc6115b45d860 | 2 | 2 | 2 | 2 | [] |
| c5db6c35771bdb0103b3f23cb94016ee | 3 | 3 | 3 | 3 | [] |
| 5bb8e842e768e7ee31473eeb3850f24f | 4 | 未进池 | 39 | 16 | [0] |
| 93bdcc1dbeff864a42755a5f95a1be7c | 5 | 13 | 5 | 4 | [] |
| cea72ade4b26344577c44d92ccb88808 | 9 | 6 | 4 | 5 | [] |

### db6bd302e70ad317ba170b061636fc0f

1911.10401.md > A Transformer-based approach to Irony and Sarcasm detection > Methodology: A hybrid Recurrent Convolution Transformer Approach > The background: Transfer Learning

Model specifications used for the state-of-the-art models compared can be found in Appendix SECREF6.


### a710fac01ff2981442bfc6115b45d860

1911.10401.md > A Transformer-based approach to Irony and Sarcasm detection > Abstract

Significantly extending our previous work [71], we propose a neural network methodology that builds on a recently proposed pre-trained transformer-based network architecture which, is further enhanced with the employment and devise of a recurrent convolutional neural network (RCNN). With this set-up, data preprocessing is kept in minimum. The performance of the devised hybrid neural architecture is tested on four benchmark datasets, and contrasted with other relevant state of the art methodologies and systems. Results demonstrate that the proposed methodology achieves state of the art performance under all benchmark datasets, outperforming, even by a large margin, all other methodologies and published studies.


### c5db6c35771bdb0103b3f23cb94016ee

1911.10401.md > A Transformer-based approach to Irony and Sarcasm detection > Conclusion

In this study, we propose the first transformer based methodology, leveraging the pre-trained RoBERTa model combined with a recurrent convolutional neural network, to tackle figurative language in social media. Our network is compared with all, to the best of our knowledge, published approaches under four different benchmark dataset. In addition, we aim to minimize preprocessing and engineered feature extraction steps which are, as we claim, unnecessary when using overly trained deep learning methods such as transformers. In fact, hand crafted features along with preprocessing techniques such as stemming and tagging on huge datasets containing thousands of samples are almost prohibited in terms of their computation cost.


### 5bb8e842e768e7ee31473eeb3850f24f

1911.10401.md > A Transformer-based approach to Irony and Sarcasm detection > Experimental Results

To assess the performance of the proposed method we performed an exhaustive comparison with several advanced state-of-the-art methodologies along with published results. The used methodologies were appropriately implemented using the available codes and guidelines, and include: ELMo BIBREF64, USE BIBREF79, NBSVM BIBREF93, FastText BIBREF94, XLnet base cased model (XLnet) BIBREF88, BERT BIBREF18 in two setups: BERT base cased (BERT-Cased) and BERT base uncased (BERT-Uncased) models, and RoBERTa base model. The published results were acquired from the respective original publication (the reference publication is indicated in the respective tables). For the comparison we utilized benchmark datasets that include ironic, sarcastic and metaphoric expressions.


### 93bdcc1dbeff864a42755a5f95a1be7c

1911.10401.md > A Transformer-based approach to Irony and Sarcasm detection > Experimental Results

Finally, Table TABREF17 compares the results from baseline models, from top two ranked task participants BIBREF68, BIBREF67, from our previous study with the DESC methodology BIBREF0 with the proposed RCNN-RoBERTa framework on a Sentiment Analysis task with figurative language, using the SemEval 2015 Task 11 dataset.


### cea72ade4b26344577c44d92ccb88808

1911.10401.md > A Transformer-based approach to Irony and Sarcasm detection > Methodology: A hybrid Recurrent Convolution Transformer Approach > Proposed Method - Recurrent CNN RoBERTA (RCNN-RoBERTa)

The intuition behind our proposed RCNN-RoBERTa approach is founded on the following observation: as pre-trained networks are beneficial for several down-streaming tasks, their outputs could be further enhanced if processed properly by other networks. Towards this end, we devised an end-to-end model with minimum training time that utilizes pre-trained RoBERTa weights combined with a RCNN in order to capture contextual information. Actually, the proposed leaning model is based on a hybrid DL neural architecture that utilizes pre-trained transformer models and feed the hidden representations of the transformer into a Recurrent Convolutional Neural Network (RCNN), similar to BIBREF92.

### 金标准证据

证据 0

To assess the performance of the proposed method we performed an exhaustive comparison with several advanced state-of-the-art methodologies along with published results. The used methodologies were appropriately implemented using the available codes and guidelines, and include: ELMo BIBREF64, USE BIBREF79, NBSVM BIBREF93, FastText BIBREF94, XLnet base cased model (XLnet) BIBREF88, BERT BIBREF18 in two setups: BERT base cased (BERT-Cased) and BERT base uncased (BERT-Uncased) models, and RoBERTa base model. The published results were acquired from the respective original publication (the reference publication is indicated in the respective tables). For the comparison we utilized benchmark datasets that include ironic, sarcastic and metaphoric expressions. Namely, we used the dataset provided in “Semantic Evaluation Workshop Task 3” (SemEval-2018) that contains ironic tweets BIBREF95; Riloff’s high quality sarcastic unbalanced dataset BIBREF96; a large dataset containing political comments from Reddit BIBREF97; and a SA dataset that contains tweets with various FL forms from “SemEval-2015 Task 11” BIBREF66. All datasets are used in a binary classification manner (i.e., irony/sarcasm vs. literal), except from the “SemEval-2015 Task 11” dataset where the task is to predict a sentiment integer score (from -5 to 5) for each tweet (refer to BIBREF0 for more details). The evaluation was made across standard five metrics namely, Accuracy (Acc), Precision (Pre), Recall (Rec), F1-score (F1), and Area Under the Receiver Operating Characteristics Curve (AUC). For the SA task the cosine similarity metric (Cos) and mean squared error (MSE) metrics are used, as proposed in the original study BIBREF66.

## 1805.04579-q03

Which combination of semantic and statistical models performs best on multi-document extractive summarization in terms of ROUGE score?

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
| 5e940db0db50eec76b6f274e6b01df29 | 1 | 未进池 | 26 | 10 | [0] |
| 1362df8315e97204758e69f38e0b36c0 | 2 | 1 | 1 | 1 | [] |
| 3beacfbb1b818a8ce3d200b911397ff1 | 3 | 24 | 5 | 3 | [] |
| 6f2447b8f1edda55514e93d9ce960408 | 4 | 45 | 10 | 6 | [] |
| e782f3a46253b285032ec50c2066a41a | 5 | 2 | 2 | 2 | [] |
| 5c366a1edacc931e3c644eab8b64a156 | 9 | 9 | 3 | 5 | [] |
| 06e24c7d0f18299e0ce38ab32b3dbaea | 8 | 11 | 4 | 4 | [] |

### 5e940db0db50eec76b6f274e6b01df29

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Experiments

In the Table 1, we try different model pairs with weights trained on corpus for Task 2. We have displayed mean ROUGE-2 scores for base Models. We have calculated final scores taking into consideration all normalizations, stemming, lemmatizing and clustering techniques, and the ones providing best results were used. We generally expected WordNet, Glove based semantic models to perform better given they better capture crux of the sentence and compute similarity using the same, but instead, they performed average. This is attributed to the fact they assigned high similarity scores to not so semantically related sentences. We also observe that combinations with TF/IDF and Similarity Matrices(Jaccard/Cosine) offer nearly same results. The InferSent based Summarizer performed exceptionally well.


### 1362df8315e97204758e69f38e0b36c0

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Abstract

For Statistical models, we have used TF/IDF, TextRAnk, Jaccard/Cosine Similarities. For Semantic Models, we have used WordNet-based Model and proposed two models based on Glove Vectors and Facebook's InferSent. We tested our approach on DUC 2004 dataset, generating 100-word summaries. We have discussed the system, algorithms, analysis and also proposed and tested possible improvements. ROUGE scores were used to compare to other summarizers.


### 3beacfbb1b818a8ce3d200b911397ff1

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Experiments

We evaluate our approaches on 2004 DUC(Document Understanding Conferences) dataset(https://duc.nist.gov/). The Dataset has 5 Tasks in total. We work on Task 2. It (Task 2) contains 50 news documents cluster for multi-document summarization. Only 665-character summaries are provided for each cluster. For evaluation, we use ROGUE, an automatic summary evaluation metric. It was firstly used for DUC 2004 data-set. Now, it has become a benchmark for evaluation of automated summaries. ROUGE is a correlation metric for fixed-length summaries populated using n-gram co-occurrence. For comparison between model summary and to-be evaluated summary, separate scores for 1, 2, 3, and 4-gram matching are kept. We use ROUGE-2, a bi-gram based matching technique for our task.


### 6f2447b8f1edda55514e93d9ce960408

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Related Work

There is a void of hybrid summarizers; there haven't been many studies made in the area.Wong BIBREF13 conducted some preliminary research but there isn't much there on benchmark tests to our knowledge. We use a mixture of statistical and semantic models, assign weights among them by training on field-specific corpora. As there is a significant variation in choices among different fields. We support our proposal with expectations that shortcomings posed by one model can be filled with positives from others. We deploy experimental analysis to test our proposition.


### e782f3a46253b285032ec50c2066a41a

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Abstract

We report a series of experiments with different semantic models on top of various statistical models for extractive text summarization. Though statistical models may better capture word co-occurrences and distribution around the text, they fail to detect the context and the sense of sentences /words as a whole. Semantic models help us gain better insight into the context of sentences. We show that how tuning weights between different models can help us achieve significant results on various benchmarks. Learning pre-trained vectors used in semantic models further, on given corpus, can give addition spike in performance. Using weighing techniques in between different statistical models too further refines our result.


### 5c366a1edacc931e3c644eab8b64a156

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Proposed Approach

For Statistical analysis we use Similarity matrices, word co-occurrence/ n-gram model, andTF/IDF matrix. For semantic analysis we use custom Glove based model, WordNet based Model and Facebook InferSent BIBREF4 based Model. For Multi-Document Summarization,after training on corpus, we assign weights among the different techniques .We store the sense vector for documents, along with weights, for future reference. For Single document summarization, firstly we calculate the sense vector for that document and calculate the nearest vector from the stored Vectors, we use the weights of the nearest vector. We will describe the flow for semantic and statistical models separately.


### 06e24c7d0f18299e0ce38ab32b3dbaea

1805.04579.md > Using Statistical and Semantic Models for Multi-Document Summarization > Introduction

We have also discussed how we can effectively subdue shortcomings of one model by using it in coalition with models which capture the view that other faintly held.

### 金标准证据

证据 0

In the Table 1, we try different model pairs with weights trained on corpus for Task 2. We have displayed mean ROUGE-2 scores for base Models. We have calculated final scores taking into consideration all normalizations, stemming, lemmatizing and clustering techniques, and the ones providing best results were used. We generally expected WordNet, Glove based semantic models to perform better given they better capture crux of the sentence and compute similarity using the same, but instead, they performed average. This is attributed to the fact they assigned high similarity scores to not so semantically related sentences. We also observe that combinations with TF/IDF and Similarity Matrices(Jaccard/Cosine) offer nearly same results. The InferSent based Summarizer performed exceptionally well. We initially used pre-trained features to generate sentence vectors through InferSent.

## 1804.09692-q03

How is instability defined for word embeddings?

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
| 3650ca3c5d86172f6c7ee7c045af0845 | 1 | 22 | 7 | 2 | [] |
| 9d3359290440dd103eb84ad6feccb178 | 2 | 47 | 17 | 7 | [0] |
| 60706d1e127aa58399a54333d4a528be | 3 | 1 | 1 | 1 | [] |
| e930678088a007310bcf2c830ee9361f | 4 | 7 | 2 | 3 | [] |
| 04463549f21905ed5c8bd6e0c02e16da | 5 | 17 | 9 | 5 | [] |
| 7ab7cc655e19df1acc723cb5da634c94 | 7 | 5 | 3 | 4 | [] |
| feee61cd4aaa3e1f496bea10f9204cf0 | 10 | 4 | 4 | 8 | [] |
| 3c3a0ccef884d9045c1be46f45d944a6 | 11 | 6 | 5 | 10 | [] |

### 3650ca3c5d86172f6c7ee7c045af0845

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Introduction

Word embeddings are low-dimensional, dense vector representations that capture semantic properties of words. Recently, they have gained tremendous popularity in Natural Language Processing (NLP) and have been used in tasks as diverse as text similarity BIBREF0 , part-of-speech tagging BIBREF1 , sentiment analysis BIBREF2 , and machine translation BIBREF3 . Although word embeddings are widely used across NLP, their stability has not yet been fully evaluated and understood. In this paper, we explore the factors that play a role in the stability of word embeddings, including properties of the data, properties of the algorithm, and properties of the words. We find that word embeddings exhibit substantial instabilities, which can have implications for downstream tasks.


### 9d3359290440dd103eb84ad6feccb178

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Defining Stability

This definition of stability can be generalized to more than two embedding spaces by considering the average overlap between two sets of embedding spaces. Let INLINEFORM12 and INLINEFORM13 be two sets of embedding spaces. Then, for every pair of embedding spaces INLINEFORM14 , where INLINEFORM15 and INLINEFORM16 , take the ten nearest neighbors of INLINEFORM17 in both INLINEFORM18 and INLINEFORM19 and calculate percent overlap. Let the stability be the average percent overlap over every pair of embedding spaces INLINEFORM20 .


### 60706d1e127aa58399a54333d4a528be

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Introduction

In the following experiments, we explore which factors affect stability, as well as how this stability affects downstream tasks that word embeddings are commonly used for. To our knowledge, this is the first study comprehensively examining the factors behind instability.


### e930678088a007310bcf2c830ee9361f

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Introduction

Using the overlap between nearest neighbors in an embedding space as a measure of stability (see sec:definingStability below for more information), we observe that many common embedding spaces have large amounts of instability. For example, Figure FIGREF1 shows the instability of the embeddings obtained by training word2vec on the Penn Treebank (PTB) BIBREF4 . As expected, lower frequency words have lower stability and higher frequency words have higher stability. What is surprising however about this graph is the medium-frequency words, which show huge variance in stability. This cannot be explained by frequency, so there must be other factors contributing to their instability.


### 04463549f21905ed5c8bd6e0c02e16da

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Impact of Stability on Downstream Tasks

Word embeddings are used extensively as the first stage of neural networks throughout NLP. Typically, embeddings are initalized based on a vector trained with word2vec or GloVe and then further modified as part of training for the target task. We study two downstream tasks to see whether stability impacts performance.

Since we are interested in seeing the impact of word vector stability, we choose tasks that have an intuitive evaluation at the word level: word similarity and POS tagging.


### 7ab7cc655e19df1acc723cb5da634c94

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Abstract

Despite the recent popularity of word embedding methods, there is only a small body of work exploring the limitations of these representations. In this paper, we consider one aspect of embedding spaces, namely their stability. We show that even relatively high frequency words (100-200 occurrences) are often unstable. We provide empirical evidence for how various factors contribute to the stability of word embeddings, and we analyze the effects of stability on downstream tasks.


### feee61cd4aaa3e1f496bea10f9204cf0

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Related Work

Finally, embeddings can be analyzed using second-order properties of embeddings (e.g., how a word relates to the words around it). Newman-Griffis and Fosler-Lussier ( BIBREF15 ) validate the usefulness of second-order properties, by demonstrating that embeddings based on second-order properties perform as well as the typical first-order embeddings. Here, we use second-order properties of embeddings to quantify stability.


### 3c3a0ccef884d9045c1be46f45d944a6

1804.09692.md > Factors Influencing the Surprising Instability of Word Embeddings > Related Work

There has been much recent interest in the applications of word embeddings, as well as a small, but growing, amount of work analyzing the properties of word embeddings.

### 金标准证据

证据 0

We define stability as the percent overlap between nearest neighbors in an embedding space. Given a word INLINEFORM0 and two embedding spaces INLINEFORM1 and INLINEFORM2 , take the ten nearest neighbors of INLINEFORM3 in both INLINEFORM4 and INLINEFORM5 . Let the stability of INLINEFORM6 be the percent overlap between these two lists of nearest neighbors. 100% stability indicates perfect agreement between the two embedding spaces, while 0% stability indicates complete disagreement. In order to find the ten nearest neighbors of a word INLINEFORM7 in an embedding space INLINEFORM8 , we measure distance between words using cosine similarity. This definition of stability can be generalized to more than two embedding spaces by considering the average overlap between two sets of embedding spaces. Let INLINEFORM12 and INLINEFORM13 be two sets of embedding spaces. Then, for every pair of embedding spaces INLINEFORM14 , where INLINEFORM15 and INLINEFORM16 , take the ten nearest neighbors of INLINEFORM17 in both INLINEFORM18 and INLINEFORM19 and calculate percent overlap. Let the stability be the average percent overlap over every pair of embedding spaces INLINEFORM20 .

## 1907.12984-q02

Is the context-aware simultaneous interpreting model compared against a system that does not use streaming text but has the entire text available?

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
| 1bf58435e70f30b9b270867f533de2aa | 1 | 6 | 3 | 1 | [] |
| a9594bb4945c7c61c65d12ec0076be8e | 2 | 3 | 2 | 2 | [] |
| c2f8a5dcf145c8f6fa4819d9a3d93e58 | 3 | 45 | 17 | 7 | [1] |
| cd0d3772bf5e4c539e6b14869b3e01e3 | 4 | 1 | 1 | 3 | [] |
| 63972cc6ae8de9a8a0b792bf543fb0bc | 5 | 37 | 16 | 9 | [] |
| 2b610559efc8b0e386cef603d051ddc9 | 9 | 4 | 4 | 6 | [] |
| 8c4977b46c5a4d26b22ba8462da64f8b | 8 | 5 | 5 | 5 | [] |

### 1bf58435e70f30b9b270867f533de2aa

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Introduction

To make the simultaneous machine translation more accessible and producible, we borrow SI strategies used by human interpreters to create our model. As shown in Figure FIGREF3 , this model is able to constantly read streaming text from the ASR model, and simultaneously determine the boundaries of Information Units (IUs) one after another. Each detected IU is then translated into a fluent translation with two simple yet effective decoding strategies: partial decoding and context-aware decoding. Specifically, IUs at the beginning of each sentence are sent to the partial decoding module. Other information units, either appearing in the middle or at the end of a sentence, are translated into target language by the context-aware decoding module.


### a9594bb4945c7c61c65d12ec0076be8e

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Abstract

In this paper, we present DuTongChuan, a novel context-aware translation model for simultaneous interpreting. This model allows to constantly read streaming text from the Automatic Speech Recognition (ASR) model and simultaneously determine the boundaries of Information Units (IUs) one after another. The detected IU is then translated into a fluent translation with two simple yet effective decoding strategies: partial decoding and context-aware decoding. In practice, by controlling the granularity of IUs and the size of the context, we can get a good trade-off between latency and translation quality easily.


### c2f8a5dcf145c8f6fa4819d9a3d93e58

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Experiments

In our context-sensitive model, the dynamic context based information unit boundary detector is essential to determine the IU boundaries in the steaming input. To measure the effectiveness of this model, we compare its precision as well as latency against the traditional language model based methods, a 5-gram language model trained by KenLM toolkit , and an in-house implemented RNN based model. Both of two contrastive models are trained on approximate 2 million monolingual Chinese sentences. As shown in Table TABREF60 , it is clear that our model beats the previous work with an absolute improvement of more than 15 points in term of F-score (62.79 INLINEFORM0 78.26) and no obvious burden in latency (average latency).


### cd0d3772bf5e4c539e6b14869b3e01e3

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Conclusion and Future Work

In this paper, we propose DuTongChuan, a novel context-aware translation model for simultaneous interpreting. This model is able to constantly read streaming text from the ASR model, and simultaneously determine the boundaries of information units one after another. The detected IU is then translated into a fluent translation with two simple yet effective decoding strategies: partial decoding and context-aware decoding. We also release a novel speech translation corpus, BSTC, to boost the research on robust speech translation task.


### 63972cc6ae8de9a8a0b792bf543fb0bc

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Related Work

However, the cascaded model remains the dominant approach and presents superior performance practically, since the ASR and NMT model can be optimized separately training on the large-scale corpus.


### 2b610559efc8b0e386cef603d051ddc9

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Conclusion and Future Work

With elaborate comparison, our model obtains superior translation quality against the wait-k model, but also presents competitive performance in latency. Assessment from human translators reveals that our system achieves promising translation quality (85.71% for Chinese-English, and 86.36% for English-Chinese), specially in the sense of surprisingly good discourse coherence. Our system also presents superior performance in latency (delayed in less 3 seconds at most times) in a speech-to-speech simultaneous translation. We also deploy our simultaneous machine translation model in our AI platform, and welcome the other users to enjoy it.

In the future, we will conduct research on novel method to evaluate the interpreting.


### 8c4977b46c5a4d26b22ba8462da64f8b

1907.12984.md > DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting > Experiments

As shown in Table TABREF70 , the performance of our model is comparable to the interpreting. It is worth mentioning that both automatic and human evaluation criteria are designed for evaluating written translation and have a special emphasis on adequacy and faithfulness. But in simultaneous interpreting, human interpreters routinely omit less-important information to overcome their limitations in working memory. As the last column in Table 6 shows, human interpreters' oral translations have more omissions than machine's and receive lower acceptability. The evaluation results do not mean that machines have exceeded human interpreters in simultaneous interpreting. Instead, it means we need machine translation criteria that suit simultaneous interpreting.

### 金标准证据

证据 0

For fair comparison, we implement the following models:

证据 1

In our context-sensitive model, the dynamic context based information unit boundary detector is essential to determine the IU boundaries in the steaming input. To measure the effectiveness of this model, we compare its precision as well as latency against the traditional language model based methods, a 5-gram language model trained by KenLM toolkit , and an in-house implemented RNN based model. Both of two contrastive models are trained on approximate 2 million monolingual Chinese sentences. As shown in Table TABREF60 , it is clear that our model beats the previous work with an absolute improvement of more than 15 points in term of F-score (62.79 INLINEFORM0 78.26) and no obvious burden in latency (average latency). This observation indicates that with bidirectional context, the model can learn better representation to help the downstream tasks. In the next experiments, we will evaluate models given testing data with IU boundaries detected by our detector.

证据 2

baseline: A standard Transformer based model with big version of hyper parameters.

证据 3

sub-sentence: We split a full sentence into multiple sub-sentences by comma, and translate them using the baseline model. To evaluate the translation quality, we concatenate the translation of each sub-sentence into one sentence.

## 2003.13028-q05

What is the architecture of the saliency model used in combination with pre-trained sequence-to-sequence models for abstractive summarization?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 0 | 0.0000 |
| bm25 | 1 | 0.5000 |
| rrf_a0 | 0 | 0.0000 |
| rrf_a0.25 | 0 | 0.0000 |
| rrf_a0.5 | 0 | 0.0000 |
| rrf_a1 | 1 | 0.5000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| 7b2c52b285f9df4339745b960c161250 | 1 | 11 | 5 | 1 | [] |
| d9f0b6d8455ef06f80a2bb37076e44df | 2 | 12 | 7 | 3 | [] |
| 3bd00e203ea8db09a7a27e2c5795dd46 | 3 | 4 | 1 | 2 | [] |
| c05c0274cf5b252edf548ddb4a645d03 | 4 | 9 | 6 | 5 | [] |
| 7e63e3ba33af96a12a2612d7f98709d6 | 5 | 3 | 2 | 4 | [] |
| ad31a254c4ae275807deeff48f9c5862 | 8 | 1 | 3 | 6 | [] |
| a8a97f2ea578d3d3acc01cd5e964bee6 | 9 | 2 | 4 | 7 | [1] |

### 7b2c52b285f9df4339745b960c161250

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Abstract

Pre-trained sequence-to-sequence (seq-to-seq) models have significantly improved the accuracy of several language generation tasks, including abstractive summarization. Although the fluency of abstractive summarization has been greatly improved by fine-tuning these models, it is not clear whether they can also identify the important parts of the source text to be included in the summary. In this study, we investigated the effectiveness of combining saliency models that identify the important parts of the source text with the pre-trained seq-to-seq models through extensive experiments. We also proposed a new combination model consisting of a saliency model that extracts a token sequence from a source text and a seq-to-seq model that takes the sequence as an additional input text.


### d9f0b6d8455ef06f80a2bb37076e44df

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Conclusion

While recent studies have been conducted to improve summarization accuracy by increasing the amount of pre-training data and developing new pre-training strategies, this study sheds light on the importance of saliency models in abstractive summarization.


### 3bd00e203ea8db09a7a27e2c5795dd46

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Pre-trained seq-to-seq Model

There are several pre-trained seq-to-seq models applied for abstractive summarization BIBREF7, BIBREF8, BIBREF2. The models use a simple Transformer-based encoder-decoder model BIBREF9 in which the encoder-decoder model is pre-trained on large unlabeled data.


### c05c0274cf5b252edf548ddb4a645d03

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Related Work and Discussion > Pre-trained Language Models for Abstractive Summarization

BIBREF18 used BERT for their sentence-level extractive summarization model. BIBREF19 proposed a new pre-trained model that considers document-level information for sentence-level extractive summarization. Several researchers have published pre-trained encoder-decoder models very recently BIBREF20, BIBREF1, BIBREF2. BIBREF20 pre-trained a Transformer-based pointer-generator model. BIBREF1 pre-trained a standard Transformer-based encoder-decoder model using large unlabeled data and achieved state-of-the-art results. BIBREF8 and BIBREF16 extended the BERT structure to handle seq-to-seq tasks.

All the studies above focused on how to learn a universal pre-trained model; they did not consider the combination of pre-trained and saliency models for an abstractive summarization model.


### 7e63e3ba33af96a12a2612d7f98709d6

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Related Work and Discussion > Abstractive Summarization with Saliency Models

In contrast, our purpose is to clarify whether combined models are effective or not, and we are the first to investigate the combination of pre-trained seq-to-seq and saliency models. We compared a variety of combinations and clarified which combination is the most effective.


### ad31a254c4ae275807deeff48f9c5862

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Combined Models

This section describes nine combinations of the pre-trained seq-to-seq model and saliency models.


### a8a97f2ea578d3d3acc01cd5e964bee6

2003.13028.md > Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models > Saliency Models > Two Types of Saliency Model for Combination

In this study, we use two types of saliency model for combination: a shared encoder and an extractor. Each model structure is based on the basic saliency model. We describe them below.

### 金标准证据

证据 0

A basic saliency model consists of $M$-layer Transformer encoder blocks ($\mathrm {Encoder}_\mathrm {sal}$) and a single-layer feed-forward network. We define the saliency score of the $l$-th token ($1 \le l \le L$) in the source text as

证据 1

In this study, we use two types of saliency model for combination: a shared encoder and an extractor. Each model structure is based on the basic saliency model. We describe them below.

证据 2

The encoder block consists of a self-attention module and a two-layer feed-forward network.

## 1909.10481-q04

What baseline is used for zero-shot cross-lingual natural language generation with a pretrained sequence-to-sequence model?

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| dense | 0 | 0.0000 |
| bm25 | 0 | 0.0000 |
| rrf_a0 | 0 | 0.0000 |
| rrf_a0.25 | 1 | 1.0000 |
| rrf_a0.5 | 1 | 1.0000 |
| rrf_a1 | 1 | 1.0000 |

| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |
|---|---|---|---|---|---|
| df42d56b5fff202d9c1fd58f603bb98f | 1 | 4 | 1 | 1 | [] |
| c20b2c53f6a419a5d3d4decded39ce6e | 2 | 5 | 2 | 2 | [] |
| 3f4f0632f12a1c194c5e3bc5d7a5cb48 | 3 | 未进池 | 22 | 11 | [] |
| 7e818459ecc1bcbc0353f61e3d949a75 | 4 | 8 | 3 | 3 | [] |
| 200ae0df9fef4b77b01e163102288d59 | 5 | 未进池 | 23 | 14 | [] |
| 7901be7600f292278276c212c1261606 | 6 | 6 | 4 | 4 | [0, 1, 5, 6] |
| 7721acbf72cdd57cb7437113d1ede0d5 | 13 | 1 | 5 | 6 | [] |

### df42d56b5fff202d9c1fd58f603bb98f

1909.10481.md > Cross-Lingual Natural Language Generation via Pre-Training > Conclusion

In this paper, we propose a pre-training method for cross-lingual natural language generation (NLG) that can transfer monolingual NLG supervision signals to all pre-trained languages. With the pre-trained model, we achieve zero-shot cross-lingual NLG on several languages by only fine-tuning once. Experimental results show that our model outperforms the machine-translation-based pipeline model on several cross-lingual NLG tasks. For future work, we would like to improve our pre-training method towards the fully unsupervised setting.


### c20b2c53f6a419a5d3d4decded39ce6e

1909.10481.md > Cross-Lingual Natural Language Generation via Pre-Training > Abstract

Experimental results on question generation and abstractive summarization show that our model outperforms the machine-translation-based pipeline methods for zero-shot cross-lingual generation. Moreover, cross-lingual transfer improves NLG performance of low-resource languages by leveraging rich-resource language data. Our implementation and data are available at https://github.com/CZWin32768/xnlg.


### 3f4f0632f12a1c194c5e3bc5d7a5cb48

1808.10059.md > Zero-Shot Adaptive Transfer for Conversational Language Understanding > Baseline Systems

For comparison with a state-of-the-art zero-shot model, we implement the concept tagger (CT) BIBREF5 . The CT model consists of a single 256 dimensional bidirectional LSTM layer that takes pre-trained word embeddings as input to produce contextual word representations. This is followed by a feed forward layer where the contextual word representations are combined with a slot encoding to produce vectors of 128 dimensions. The slot encoding is the average vector of the word embeddings for the slot description. This feeds into another 128 dimensional bi-directional LSTM layer followed by a softmax layer that outputs the prediction for that slot.


### 7e818459ecc1bcbc0353f61e3d949a75

1909.10481.md > Cross-Lingual Natural Language Generation via Pre-Training > Experiments > Abstractive Summarization > Zero-Shot Summarization

In the zero-shot setting, we only use English data for training, and directly evaluate the model on other languages. In Table TABREF22 and Table TABREF23, we present the results for French/Chinese AS, which are evaluated by the ROUGE-1, ROUGE-2 and ROUGE-L metrics. We also report the results of supervised AS in Table TABREF21 for reference. We find that Xnlg outperforms all the baseline models on both French and Chinese AS. Comparing with French, there is a larger gap between baselines and our model on zero-shot Chinese AS, which indicates that the error propagation issue is more serious on distant language pairs.


### 200ae0df9fef4b77b01e163102288d59

1808.10059.md > Zero-Shot Adaptive Transfer for Conversational Language Understanding > Baseline Systems

In order to compare our method against the state-of-the-art models, we compare against the models presented in BIBREF4 , including the BoE models and their non-BoE variants. We also compare our method with another zero-shot model for slot tagging BIBREF5 in domain adaptation settings.


### 7901be7600f292278276c212c1261606

1909.10481.md > Cross-Lingual Natural Language Generation via Pre-Training > Experiments > Question Generation > English-English Question Generation

We first conduct experiments on the supervised English-English QG setting. We compare our model to the following baselines:

CorefNqg BIBREF33 A sequence-to-sequence model with attention mechanism and a feature-rich encoder.

Mp-Gsn BIBREF31 A sequence-to-sequence model with gated self-attention and maxout pointer mechanism.

Xlm BIBREF5 The current state-of-the-art cross-lingual pre-training model. We initialize the Transformer-based sequence-to-sequence model with pre-trained XLM.

We evaluate models with BLEU-4 (BL-4), ROUGE (RG) and METEOR (MTR) metrics. As shown in Table TABREF16, our model outperforms the baselines, which demonstrates that our pre-trained model provides a good initialization for NLG.


### 7721acbf72cdd57cb7437113d1ede0d5

1909.10481.md > Cross-Lingual Natural Language Generation via Pre-Training > Abstract

In this work we focus on transferring supervision signals of natural language generation (NLG) tasks between multiple languages. We propose to pretrain the encoder and the decoder of a sequence-to-sequence model under both monolingual and cross-lingual settings. The pre-training objective encourages the model to represent different languages in the shared space, so that we can conduct zero-shot cross-lingual transfer. After the pre-training procedure, we use monolingual data to fine-tune the pre-trained model on downstream NLG tasks. Then the sequence-to-sequence model trained in a single language can be directly evaluated beyond that language (i.e., accepting multi-lingual input and producing multi-lingual output).

### 金标准证据

证据 0

CorefNqg BIBREF33 A sequence-to-sequence model with attention mechanism and a feature-rich encoder.

证据 1

Mp-Gsn BIBREF31 A sequence-to-sequence model with gated self-attention and maxout pointer mechanism.

证据 2

Pipeline (Xlm) The pipeline of translating input Chinese sentences into English first, then performing En-En-QG with the XLM model, and finally translating back to the Chinese. We use the Transformer as the translator, which is also trained on the MultiUN dataset.

证据 3

Pipeline (Xlm) with Google Translator Same to Pipeline (Xlm) but using Google Translator to translate the texts.

证据 4

We conduct experiments on the zero-shot Chinese-Chinese QG task to evaluate the cross-lingual transfer ability. In this task, models are trained with English QG data but evaluated with Chinese QG examples. We include the following models as our baselines:

证据 5

We first conduct experiments on the supervised English-English QG setting. We compare our model to the following baselines:

证据 6

Xlm BIBREF5 The current state-of-the-art cross-lingual pre-training model. We initialize the Transformer-based sequence-to-sequence model with pre-trained XLM.

证据 7

Xlm Fine-tuning XLM with the English QG data.

