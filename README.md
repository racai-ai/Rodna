## RODNA
**RO**manian **D**eep **N**eural networks **A**rchitectures (RODNA) is a Python 3/Tensorflow/Keras project with the declared goal of obtaining better results at Romanian text processing through the use of Romanian-specific features than generic, language-independent ML toolkits.

**The project is under development.** Check back regularly for a stable version.


## Sentence splitter
10 epochs

train set: loss: 2.5142e-04 - categorical_accuracy: 0.9999

on_epoch_end: SENTEND dev precision at epoch 10 is P = 0.9959

on_epoch_end: SENTEND dev recall at epoch 10 is R = 0.9938

on_epoch_end: SENTEND dev f-measure at epoch 10 is F1 = 0.9949

_train_keras_model: SENTEND test Precision is P = 0.9963

_train_keras_model: SENTEND test Recall is R = 0.9936

_train_keras_model: SENTEND test F-measure is F1 = 0.995


## Romanian morphology
A RNN neural network than learns the mapping from a word form to its possible morpho-syntactic labels.

50 epochs

train set: loss: 2.0254e-04 - true_positives: 506802.7682 - precision: 0.9799 - recall: 0.9739

dev set: val_loss: 7.7582e-04 - val_true_positives: 107274.0000 - val_precision: 0.9389 - val_recall: 0.9318


## POS Tagger
To be continued...
