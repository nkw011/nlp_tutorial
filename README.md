# NLP Review & Tutorial

NLP 실습용 GitHub Repository 입니다.

Blog: [velog](https://velog.io/@nkw011), [GitHub Blog](https://nkw011.github.io)

## Huggingface Usage
### Tutorial
[Transformers Tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial)의 내용을 한국어 데이터셋을 이용해 공부할 수 있도록 새롭게 구성하여 만들었습니다.

* Huggingface 설치: 내용, 실습
* Tutorial1 - Pipelines for inference: 내용, 실습 
* Tutorial2 - Load pretrained instances with an AutoClass: 내용, 실습
* Tutorial3 - Preprocess: 내용, 실습
* Tutorial4 - Fine-tune a pretrained model: 내용, 실습

## Data Preprocessing

* Tokenization: NLTK, spaCy, torchtext 라이브러리를 활용해 토큰화를 수행합니다.
  * [실습](https://github.com/nkw011/nlp_tutorial/blob/main/data_preprocessing/Tokenizer.ipynb)
* Vocab: spaCy, torchtext를 활용해 Vocab을 만듭니다.
  * [실습](https://github.com/nkw011/nlp_tutorial/blob/main/data_preprocessing/Vocab.ipynb)
* Dataset & DataLoader: NLP Task를 위한 Dataset, DataLoader을 만듭니다.
  * [실습](https://github.com/nkw011/nlp_tutorial/blob/main/data_preprocessing/Dataset_Dataloader.ipynb)

## Model

* RNN, LSTM
  * [RNN 정리](https://velog.io/@nkw011/rnn)
  * [LSTM, GRU 정리](https://velog.io/@nkw011/lstm-gru)
  * [실습](https://github.com/nkw011/nlp_tutorial/blob/main/RNN_LSTM/Language_Modeling.ipynb)
* Seq2Seq
  * [Seq2Seq 정리](https://velog.io/@nkw011/seq-to-seq)
  * [실습](https://github.com/nkw011/nlp_tutorial/blob/main/Seq2Seq/NMT_with_seq2seq.ipynb)
* Transformer
  * [Transformer 정리](https://velog.io/@nkw011/transformer)
  * [실습](https://github.com/nkw011/nlp_tutorial/blob/main/Transformer/NMT_with_Transformer.ipynb)