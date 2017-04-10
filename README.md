# Machine Translation Quality Estimation

Training and inference code from the paper [A Recurrent Neural Networks Approach for Estimating the Quality of Machine Translation Output](http://www.aclweb.org/anthology/N16-1059)

There are three subdirectories:
### qualvec
Using a parallel corpora, trains the neural machine translation (encoder-decoder-attention) model  with two modifications 1) Both the encoder and decoder are bidirectional 2) introduce another hidden layer between maxout and the projection matrix in the decoder. During inference, instead of predicting the next word, we take the element-wise multipication of the row vector from the projection matrix of the  translated word and the additional hidden layer's node values from prediction to create an array of quality vectors (See paper for more details). Training data for quality estimation are then run through this model to obtain the quality vectors.

### qescore
The 2nd step to training the quality estimation model. Takes the quality vector from qualvec, runs a GRU RNN on it and use last hidden state to predict the quality vector. Currently uses L2 loss, which is unstable, but should modify to use loss function from this [paper](https://github.com/stanfordnlp/treelstm) for better stability.

### paradet
Similar to qualvec except uses "general" attention from [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) which worked better. Will merge with qualvec.


To Train:

Tokenize text with the [moses tokenizer](http://www.statmt.org/moses/?n=moses.baseline)

For training NMT model:
'''
python qualvec/qualvec.py --data_dir <data directory containing parallel corpora> --train_dir <file path to directory storing trained model and log files>

For obtaining quality vectors:
python qescore/qe.py --data_dir <data directory containing quality estimation data> --train_dir <training dir>
'''