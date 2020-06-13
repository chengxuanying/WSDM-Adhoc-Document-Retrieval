## WSDM-Adhoc-Document-Retrieval

This is our solution for [WSDM - DiggSci 2020](http://www.wsdm-conference.org/2020/wsdm-cup-2020.php). We implemented a simple yet robust search pipeline which ranked 2nd in the validation set and 4th in the test set. We won the gold prize at innovation track and bronze prize at dataset track. [[Video](https://vimeo.com/389698304)] [[Slides](https://drive.google.com/open?id=1gr3Trg_3uO0c2waUKdEz9sQUKT9Agdv06dFLqS2t4zw)] [[Report](http://www.wsdm-conference.org/2020/wsdm_cup_reports/Task1_dlutycx.pdf)] 

##### Related Project: [KDD-Multimodalities-Recall](https://github.com/chengsyuan/KDD-Multimodalities-Recall)
### Features

* An end-to-end system with **zero feature engineering**.
* Performed data cleaning on the dataset according to self-designed saliency-based rules, and removed the redundancy data with an insignificant impact on results, and improved the MAP@3 by 3%. 
* Designed a novel early stopping strategy for reranking based on the confidence score to **avoid up to 40% unnecessary inference computation** cost of the BERT.
* Scores are stable (nearly the same) on the train_val, validation, and test sets.

### Our Pipeline

1. Open the jupyter notebook ```jupyter lab or jupyter notebook```
2. Clean the dataset: ```Open 01WASH.ipynb and run all cells```. In this notebook, we clean the dataset, by removing the description text which is not highly related to the query topic. We also remove the NA rows in the candidate set. Such practice **decreases the recall size from 838,939 to 636,439** without sacrificing so much recall rate.
3. Recall: ```Open 02RECALL.ipynb and run all cells```. In this notebook, we use the bm25 metric, which is a kind of scoring method to recall the documents by the same important keywords. For faster calculating, we adopted the cupyx to accelerate the calculation. Hence, the matrix multiplication of (validsize, vocab_size) dot (600k, vocab_size) can be done in 15 mins in a single GPU card.
4. Rerank: ```Open 05BERT_ADARERANK.ipynb and run all cells```. In this notebook, we used the fintuned BioBERT model to scoring every (query, document) pair. A novel early stopping strategy is designed for saving the computation. That is, when reranking documents for a given query, if a document is scored with high confidence (above a threshold), the reranking process for this query can be earlier stopped. 

**How to finetune the BioBERT for *Scoring*:**

As mentioned in the step4, we need to finetune the BioBERT for the reranking task. Please refer to the notebook file```03BERT_PREPARE.ipynb and 04BERT_TRAIN.ipynb``` for coding details. Also, we will give some worth-mentioning tips:

1. **Use pairwise BERT**: When using Bert to scoring sentence pairs, using the [token] vector as output and followed by a single-layer neural network with dropout is recommended.
2. **Use RankNet loss**: Cross-entropy is not the best choice for the ranking problem, because it aims to train the scoring function to be inf or -inf. Such loss benefits to the classification task, while in ranking task, we do not need extreme scores. What we need is more discriminative scores -  the document more related scores higher. That is the Ranknet loss. Limited to the GPU resource, our team can not implement RankNet loss in BERT. Instead, we selected the finetuned models performing well in ranking task, which is so-called underfitting model in the classification tasks. Such practice improves 0.03+ MAP@3 in the validation set.
3. **Use 512 Tokens in Training**: For both training and inference phase, longer token means that the model can capture more semantic information. In our test, increasing token length from 256 to 512 can improve 0.02+ MAP@3.
4. **Upsample Positive Items**: Similar to the classification task, you can upsample the positive (query, doc) pairs or reweight them in the loss item.

### Members

1. **Chengxuan Ying**, Dalian University of Technology (应承轩 大连理工大学)

2. Chen Huo```Server sponsor```, Wechat (霍晨 微信)

### DataLeak

We did not use any data leak tricks, though we know the data leak exists.

### Acknownledgment

Thanks for [Yanming Shen](http://faculty.dlut.edu.cn/yshen/zh_CN/index.htm), who provided a 8-GPU server for 4 days.

### Links to Other Solutions

* Chi-Yu Yang and Kuei-Chun Huang: [WSDM_SimpleBaseline](https://github.com/steven95421/WSDM_SimpleBaseline)
* supercoderhawk: [wsdm-digg-2020](https://github.com/supercoderhawk/wsdm-digg-2020)
* shuiliwanwu: [wsdm_cup2020](https://github.com/shuiliwanwu/wsdm_cup2020)
* just4fun, greedisgood, slowdown and funny: [wsdm2020-solution](https://github.com/wsdm-Teamfunny/wsdm2020-solution)
* xiong, wzm, Yinxiang Xu, Xiaohao Xu and Yongqiang Liu: [wsdm2020_diggsci](https://github.com/xiong666/wsdm2020_diggsci)
* Seiya, eclipse, will and ferryman: [wsdm_cup_2020_solution](https://github.com/myeclipse/wsdm_cup_2020_solution)

### Reference

1. Nogueira R, Cho K. Passage Re-ranking with BERT[J]. arXiv preprint arXiv:1901.04085, 2019.
2. Burges C, Shaked T, Renshaw E, et al. Learning to rank using gradient descent[C]//Proceedings of the 22nd International Conference on Machine learning (ICML-05). 2005: 89-96.
3. Severyn A, Moschitti A. Learning to rank short text pairs with convolutional deep neural networks[C]//Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. ACM, 2015: 373-382.
