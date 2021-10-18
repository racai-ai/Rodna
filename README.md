## RODNA
**RO**manian **D**eep **N**eural networks **A**rchitectures (RODNA) is a Python 3/Tensorflow/Keras project with the declared goal of obtaining better results at Romanian text processing through the use of Romanian-specific features than generic, language-independent ML toolkits.

**The project is under development.** Check back regularly for a stable version.


## Sentence splitter
A RNN neural network that does sentence splitting.

10 epochs

loss: 3.0761e-04 - categorical_accuracy: 0.9999

on_epoch_end: SENTEND dev precision at epoch 10 is P = 0.9929

on_epoch_end: SENTEND dev recall at epoch 10 is R = 0.9973

on_epoch_end: SENTEND dev f-measure at epoch 10 is F1 = 0.9951

_train_keras_model: SENTEND test Precision is P = 0.992

_train_keras_model: SENTEND test Recall is R = 0.9958

_train_keras_model: SENTEND test F-measure is F1 = 0.9939

## Romanian morphology
A RNN neural network than learns the mapping from a word form to its possible morpho-syntactic labels.

50 epochs

loss: 1.9136e-04 - true_positives: 1014386.0000 - precision: 0.9807 - recall: 0.9756

val_loss: 7.7358e-04 - val_true_positives: 107891.0000 - val_precision: 0.9421 - val_recall: 0.9334

## POS Tagger
To be continued...
