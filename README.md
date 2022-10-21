# Introduction

The Self-Supervised-QA repository is the PyTorch Implementation of COLING 2022 Paper [Domain Adaptation for Question Answering via Question Classification](https://arxiv.org/abs/2209.04998) and EMNLP 2022 Paper [QA Domain Adaptation using Hidden Space Augmentation and Self-Supervised Contrastive Adaptation](https://arxiv.org/abs/2210.10861)

In [Domain Adaptation for Question Answering via Question Classification](https://arxiv.org/abs/2209.04998),  we investigate the potential benefits of question classification for QA domain adaptation. We propose a novel framework: Question Classification for Question Answering (QC4QA). Specifically, a question classifier is adopted to assign question classes to both the source and target data. Then, we perform joint training in a self-supervised fashion via pseudo-labeling. For optimization, inter-domain discrepancy between the source and target domain is reduced via maximum mean discrepancy (MMD) distance. We additionally minimize intra-class discrepancy among QA samples of the same question class for fine-grained adaptation performance. To the best of our knowledge, this is the first work in QA domain adaptation to leverage question classification with self-supervised adaptation. We demonstrate the effectiveness of the proposed QC4QA with consistent improvements against the state-of-the-art baselines on multiple datasets
<img src=pics/intro1.png>

In [QA Domain Adaptation using Hidden Space Augmentation and Self-Supervised Contrastive Adaptation](https://arxiv.org/abs/2210.10861), we propose a novel self-supervised framework called QADA for QA domain adaptation. QADA introduces a novel data augmentation pipeline used to augment training QA samples. Different from existing methods, we enrich the samples via hidden space augmentation. For questions, we introduce multi-hop synonyms and sample augmented token embeddings with Dirichlet distributions. For contexts, we develop an augmentation method which learns to drop context spans via a custom attentive sampling strategy. Additionally, contrastive learning is integrated in the proposed self-supervised adaptation framework QADA. Unlike existing approaches, we generate pseudo labels and propose to train the model via a novel attention-based contrastive adaptation method. The attention weights are used to build informative features for discrepancy estimation that helps the QA model separate answers and generalize across source and target domains. To the best of our knowledge, our work is the first to leverage hidden space augmentation and attention-based contrastive adaptation for self-supervised domain adaptation in QA. Our evaluation shows that QADA achieves considerable improvements on multiple target datasets over state-of-the-art baselines in QA domain adaptation
<img src=pics/intro2.png>
<img src=pics/intro3.png>


## Citing 

Please consider citing the following papers if you use our methods in your research:
```
@inproceedings{yue2022domain,
  title={Domain Adaptation for Question Answering via Question Classification},
  author={Yue, Zhenrui and Zeng, Huimin and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={1776--1790},
  year={2022}
}

@inproceedings{yue2022qa,
  title={QA Domain Adaptation using Hidden Space Augmentation and Self-Supervised Contrastive Adaptation},
  author={Yue, Zhenrui and Zeng, Huimin and Kratzwald, Bernhard and Feuerriegel, Stefan and Wang, Dong},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
```


## Data & Requirements

We released the QA datasets with question classes (via TREC question classification) and the question classification model [here](https://drive.google.com/drive/folders/10JPidO9bJwM1FyuEmlNfvaPx3loZNVt8?usp=sharing) 

Required packages: Pytorch, pytorch_pretrained_bert etc. For our running environment see requirements.txt


## Train Source QA Model on SQuAD

To perform domain adaptation with the proposed QC4QA or QADA, you need to run the supervised training on the source domain at first. An example:
```
CUDA_VISIBLE_DEVICES=0 python src/run_source.py \
--bert_model bert-base-uncased \
--do_train \
--train_file PATH_TO_SQUAD_TRAIN_FILE \
--predict_file PATH_TO_SQUAD_DEV_FILE \
--output_dir squad \
--output_model_file best_model.bin \
--logger_path squad
```

You can modified the parameters for --train_file, --predict_file, --output_dir and --output_model_file to update dataset paths, the model save path and model file name (Same below)


## Perform Question Classification with TREC (Optional)

This step is optional for QA4QA KMeans and QADA, we provide the classified datasets [here](https://drive.google.com/drive/folders/10JPidO9bJwM1FyuEmlNfvaPx3loZNVt8?usp=sharing). To perform the classification with TREC taxonomy, first train a question classifier with the [SentEval](https://github.com/facebookresearch/SentEval) package (Model state dict is provided in the link above), then use the saved the model and run `CUDA_VISIBLE_DEVICES=0 python src/qc4qa_trec.py`

Notice you may have to change the hyperparameters: PATH_SENTEVAL (SentEval root), PATH_TO_W2V (embedding root), MODEL_PATH (infersent model root), QA file path and path to your classification model (i.e., qc4qa_model.pth) in qc4qa_trec.py


## Run QC4QA TREC and KMeans (COLING 2022)

After obtain the supervised training model and the classified QA datasets, you can run adaptation with QC4QA given the classified source and target datasets

To run the QC4QA adaptation experiments with TREC supervised question classification, specify the paths to your classified QA datasets, hyperparameter (i.e., lambda_c for scaling the contrastive loss) and run the following command:
```
CUDA_VISIBLE_DEVICES=0 python src/run_qc4qa.py \
--bert_model bert-base-uncased \
--do_adaptation \
--do_lower_case \
--source_train_file PATH_TO_CLASSIFIED_SQUAD_TRAIN_FILE \
--target_train_file PATH_TO_CLASSIFIED_TARGET_TRAIN_FILE \
--target_predict_file PATH_TO_TARGET_DEV_FILE \
--input_dir squad \
--input_model_file best_model.bin \
--output_dir squad2target \
--output_model_file best_model.bin \
--logger_path squad2target \
--lambda_c LAMBDA_FOR_CONTRASTIVE_LOSS
```

To run the QC4QA adaptation experiments with KMeans unsupervised clustering, specify the dataset paths, hyperparameters (i.e., num_clusters for KMeans cluster number and lambda_c for scaling the contrastive loss) and run the following command:
```
CUDA_VISIBLE_DEVICES=0 python src/run_qc4qa_kmeans.py \
--bert_model bert-base-uncased \
--do_adaptation \
--do_lower_case \
--source_train_file PATH_TO_SQUAD_TRAIN_FILE \
--target_train_file PATH_TO_TARGET_TRAIN_FILE \
--target_predict_file PATH_TO_TARGET_DEV_FILE \
--input_dir squad \
--input_model_file best_model.bin \
--output_dir squad2target \
--output_model_file best_model.bin \
--logger_path squad2target \
--num_clusters 5 \
--lambda_c LAMBDA_FOR_CONTRASTIVE_LOSS
```


## Run QADA (EMNLP 2022)

With the supervised training model and the QA datasets, you can run the QADA adaptation experiments, specify the paths to your QA datasets, the augmentation hyperparameters (i.e., dirichlet_ratio for synonym sampling ratio, cutoff_ratio for context cutoff ratio and lambda_c for scaling the contrastive loss) and run the following command:
```
CUDA_VISIBLE_DEVICES=0 python src/run_qada.py \
--bert_model bert-base-uncased \
--do_adaptation \
--do_lower_case \
--source_train_file PATH_TO_SQUAD_TRAIN_FILE \
--target_train_file PATH_TO_TARGET_TRAIN_FILE \
--target_predict_file PATH_TO_TARGET_DEV_FILE \
--input_dir squad \
--input_model_file best_model.bin \
--output_dir squad2target \
--output_model_file best_model.bin \
--logger_path squad2target \
--dirichlet_ratio DIRICHLET_SAMPLING_RATIO \
--cutoff_ratio CONTEXT_CUTOFF_RATIO \
--lambda_c LAMBDA_FOR_CONTRASTIVE_LOSS
```


## Performance

The first two tables show the performance of our QC4QA
<img src=pics/performance1.png>
<img src=pics/performance2.png>

The last table reports the performance of our QADA
<img src=pics/performance3.png>


## Acknowledgement

During the implementation we base our code mostly on [Transformers](https://github.com/huggingface/transformers) from Hugging Face and [CASe](https://github.com/caoyu-noob/CASe) by Cao et al. Many thanks to these authors for their great work!
