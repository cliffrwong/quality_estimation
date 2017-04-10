# Machine Translation Quality Estimation

Training and inference code from the paper [A Recurrent Neural Networks Approach for Estimating the Quality of Machine Translation Output](http://www.aclweb.org/anthology/N16-1059)

There are four subdirectories:
### qualvec
Using a parallel corpora, trains the neural machine translation (encoder-decoder-attention) model  with two modifications 1) Both the encoder and decoder are bidirectional 2) introduce another hidden layer between maxout and the projection matrix in the decoder. During inference, instead of predicting the next word, we take the element-wise multiplication of the row vector from the projection matrix of the  translated word and the additional hidden layer's node values from prediction to create an array of quality vectors (See paper for more details). Training data for quality estimation are then run through this model to obtain the quality vectors as input to qescore. This code was modified from the tensorflow seq2seq example.

### qescore
The 2nd step to training the quality estimation model. Takes the quality vector from qualvec, runs a GRU RNN on it and use last hidden state to predict the HTER score. Currently uses L2 loss, which is unstable, but should modify to use loss function from this [paper](http://www.aclweb.org/anthology/P15-1150) for better stability.

### paradet
Similar to qualvec except uses "general" attention from [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) which worked better. Will merge with qualvec.

### baseline_svr
Get the baseline score using Support Vector Regression on baseline features. Hyperparameters are optimized using particle swarm optimization.

To Train:

Tokenize text with the [moses tokenizer](http://www.statmt.org/moses/?n=moses.baseline)

For training NMT model:
```
python qualvec/qualvec.py --data_dir <data directory containing parallel corpora> --train_dir <training dir>
```


For obtaining quality vectors
```
python qualvec/qualvec.py --data_dir <data directory containing quality estimation data> --train_dir <training dir> --qualvec
```

training quality estimation model 
```
python qescore/qe.py --data_dir <data directory containing quality vectors and labels> --train_dir <training dir>
```

quality estimation inference
```
python qescore/qe.py --data_dir <data directory containing training vectors> --train_dir <training dir> --qescore
```

See source code for additional options