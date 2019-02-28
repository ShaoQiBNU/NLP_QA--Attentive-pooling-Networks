# Attentive Pooling Networks

TensorFlow implementation of Attentive Pooling Networks [paper](https://arxiv.org/abs/1602.03609v1).

Welcome to post issues and pull requests.

## Prerequisites

 - Python 2.7
 - TensorFlow >= 1.8 
 
## Notes
Origin paper use one filter size in AP-CNN, but here we can use multiple filter sizes.
CNN and LSTM should share weights, but should attentive pooling matrix U share or not ?

There are many details need to be considered in implementation. 

## TODOs
1. add evaluation metrics MAP MRR
2. add other datasets
3. deal with last smaller batch for prediction, now only work for predict_batch_size=1


## References
[Attentive Pooling Networks](https://arxiv.org/abs/1602.03609v1)
