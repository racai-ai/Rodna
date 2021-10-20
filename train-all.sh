#!/bin/bash

# Delete all model files for the RoPOSTagger LM model
rm -fvr data/models/tagger/langmod/*
rm -fv data/models/tagger_unic_props.txt
rm -fv data/models/word_ids.txt
# Retrain the LM model
python3 -m rodna.tagger -lm
