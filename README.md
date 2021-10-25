## RODNA
**RO**manian **D**eep **N**eural networks **A**rchitectures (RODNA) is a Python 3/Tensorflow/Keras project with the declared goal of obtaining better results at Romanian text processing through the use of Romanian-specific features than generic, language-independent ML toolkits.

**The project is under development.** Check back regularly for a stable version.

All training is done on a machine with two NVIDIA cards with 10421 MB memory each: name: GeForce GTX 1080 Ti, compute capability: 6.1

## Sentence splitter
A RNN neural network that does sentence splitting.

10 epochs

loss: 3.0740e-04 - categorical_accuracy: 0.9999

on_epoch_end: SENTEND dev precision at epoch 10 is P = 0.9961

on_epoch_end: SENTEND dev recall at epoch 10 is R = 0.99

on_epoch_end: SENTEND dev f-measure at epoch 10 is F1 = 0.993

_train_keras_model: SENTEND test Precision is P = 0.9976

_train_keras_model: SENTEND test Recall is R = 0.9904

_train_keras_model: SENTEND test F-measure is F1 = 0.994

real    2m42.815s
user    2m51.797s
sys     0m5.996s

## Romanian morphology
A RNN neural network than learns the mapping from a word form to its possible morpho-syntactic labels.

50 epochs
loss: 2.0031e-04 - true_positives: 1013132.0000 - precision: 0.9798 - recall: 0.9741 

val_loss: 7.8529e-04 - val_true_positives: 107739.0000 - val_precision: 0.9409 - val_recall: 0.9335

real    14m45.532s
user    21m7.556s
sys     1m48.665s

## POS Tagger
To be continued...
