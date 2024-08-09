# Unified-Language-Processing-with-ERNIE-T5-and-Pegasus
In this we explore the large language models like Pegasus ,ERNIE and T5 Large
# ERNIE-2.0
Introduction
ERNIE 2.0 is a continual pre-training framework proposed by Baidu in 2019, which builds and learns incrementally pre-training tasks through constant multi-task learning. Experimental results demonstrate that ERNIE 2.0 outperforms BERT and XLNet on 16 tasks including English tasks on GLUE benchmarks and several common tasks in Chinese.

More detail: [Link](https://arxiv.org/abs/1907.12412)

Released Model Info
This released pytorch model is converted from the officially released PaddlePaddle ERNIE model and a series of experiments have been conducted to check the accuracy of the conversion.

Official PaddlePaddle ERNIE repo:[Repo](https://github.com/PaddlePaddle/ERNIE)

Pytorch Conversion repo:[Repo]( https://github.com/nghuyong/ERNIE-Pytorch)

How to use

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
model = AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")
```
Citation
```
@article{sun2019ernie20,
  title={ERNIE 2.0: A Continual Pre-training Framework for Language Understanding},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:1907.12412},
  year={2019} 
}
```
# The Framework of ERNIE 2.0
![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*gViU2JniIqa-b2JcUwIX-Q.png)

A continual pre-training framework named ERNIE 2.0 is proposed, which incrementally builds pre-training tasks and then learn pre-trained models on these constructed tasks via continual multi-task learning.

![Structure](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*BhTbK0cXxHz4JRnww2CbDw.png)

Image taken from Medium.com

The model is a standard [Transformer](https://arxiv.org/abs/1706.03762) encoder.
Task embedding is introduced to represent the characteristic of different tasks. Each task has an unique id ranging from 0 to N.

Later, [ERNIE 3.0](https://sh-tsang.medium.com/brief-review-ernie-3-0-large-scale-knowledge-enhanced-pre-training-for-language-understanding-7b0f777b19be) is also proposed.


# PEGASUS and PEGASUS-X


The PEGASUS-X model that is adapted for long input summarization.

The PEGASUS-X model is described in [Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347).

## Preparation

### Model Conversion from TF Checkpoints

To convert PEGASUS TensorFlow checkpoints for use with the Flax code, use the script 
[convert_from_pegasus_to_flax](checkpoint_conversion/convert_from_pegasus_to_flax.py).

### Model Conversion between encoder architectures

### Tokenizer

You will also need to download the tokenizer file from [here](https://storage.googleapis.com/pegasus_ckpt/c4.unigram.newline.10pct.96000.model).

### Checkpoints

The full set of available checkpoints can be found in the [PEGASUS GCS Bucket](https://console.cloud.google.com/storage/browser/pegasus_ckpt/).

We highlight the notable PEGASUS-X-specific checkpoints here:

* PEGASUS-X-base (272M):
  * [PEGASUS-X-base (adapted but not fine-tuned)](https://storage.googleapis.com/pegasus_ckpt/px/untuned/base/checkpoint_1800000)
  * [Fine-tuned on arXiv](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/arxiv_beam2_alpha1.ckpt)
  * [Fine-tuned on Big Patent](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/bigpatent.ckpt)
  * [Fine-tuned on PubMed](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/pubmed.ckpt)
  * [Fine-tuned on GovReport (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/scrolls_govreport.ckpt)
  * [Fine-tuned on QMSum (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/scrolls_qmsum.ckpt)
  * [Fine-tuned on SummScreen/FD (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/scrolls_summscreen.ckpt)
* PEGASUS-X (568M):
  * [PEGASUS-X (adapted but not fine-tuned)](https://storage.googleapis.com/pegasus_ckpt/px/untuned/large/checkpoint_1800000)
  * [Fine-tuned on arXiv](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/arxiv_beam2_alpha1.ckpt)
  * [Fine-tuned on Big Patent](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/bigpatent.ckpt)
  * [Fine-tuned on PubMed](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/pubmed.ckpt)
  * [Fine-tuned on GovReport (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/scrolls_govreport.ckpt)
  * [Fine-tuned on QMSum (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/scrolls_qmsum.ckpt)
  * [Fine-tuned on SummScreen/FD (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/scrolls_summscreen.ckpt)

  # Architecture of PEGASUS
  ![Architecture](https://www.catalyzex.com/_next/image?url=https%3A%2F%2Fai2-s2-public.s3.amazonaws.com%2Ffigures%2F2017-08-08%2Ff4061bd225b3be5b3f5b18eb1a229ce991efefeb%2F2-Figure1-1.png&w=640&q=75)

  # T5 Large
  
  T5 (Text-to-Text Transfer Transformer) is a series of large language models developed by Google AI. Introduced in 2019,T5 models are trained on a massive dataset of text and code using a text-to-text framework. The T5 models are capable of performing the text-based tasks that they were pretrained for. They can also be finetuned to perform other tasks.They have been employed in various applications, including chatbots, machine translation systems, text summarization tools, code generation, and robotics.

  To know more about it visit [github](https://github.com/google-research/text-to-text-transfer-transformer)
  
  ![image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*iJcUH1F0TmCQE5p2wQt9og.png)
  
  It was source from [source](http://jalammar.github.io/illustrated-transformer/)

  Baseline model for the T5 Framework is
  ![Image](https://miro.medium.com/v2/resize:fit:1582/format:webp/1*QOVXAn0bx8HKGrBIXAgydw.png)
  
  It was source from [paper](https://arxiv.org/pdf/1910.10683.pdf)
