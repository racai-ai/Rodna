#!/bin/bash

function train_sentence_splitter {
	echo
	echo ========== Training RoSentenceSplitter ==========
	echo

	# Delete all model files for the RoSentenceSplitter model
	rm -fv data/models/splitter/model.pt
	rm -fv data/models/splitter_feat_len.txt
	rm -fv data/models/splitter_unic_props.txt
	# Retrain the sentence splitter model
	python3 -m rodna.splitter

	echo
	echo ========== End training RoSentenceSplitter ==========
	echo
}

function train_morphology {
	echo
	echo ========== Training RoInflect ==========
	echo

	# Delete all model files for the RoInflect model
	rm -fv data/models/morphology/model.pt
	rm -fv data/models/char_ids.txt
	rm -fv data/models/unknown_aclasses.txt
	# Retrain the morphology model
	python3 -m rodna.morphology

	echo
	echo ========== End training RoInflect ==========
	echo
}

function train_pos_tagger {
	echo
	echo ========== Training RoPOSTagger ==========
	echo

	# Delete all model files for the RoPOSTagger model
	rm -fv data/models/tagger/cls/config.json
	rm -fv data/models/tagger/cls/model.pt
	rm -fv data/models/tagger/crf/config.json
	rm -fv data/models/tagger/crf/model.pt
	rm -fv data/models/tagger/cls_bert/*.json
	rm -fv data/models/tagger/cls_bert/model.safetensors
	rm -fv data/models/tagger/cls_bert/vocab.txt
	rm -fv data/models/tagger/crf_bert/*.json
	rm -fv data/models/tagger/crf_bert/model.safetensors
	rm -fv data/models/tagger/crf_bert/vocab.txt
	rm -fv data/models/tagger_unic_props.txt
	# Retrain the tagger model
	python3 -m rodna.tagger

	echo
	echo ========== End training RoPOSTagger ==========
	echo
}

function train_dep_parser {
	echo
	echo ========== Training RoDepParser ==========
	echo

	# Model files for the RoDepParser models 1 and 2 are automatically deleted
	# and the best model is saved
	
	# Retrain the parser model
	python3 -m rodna.parser

	echo
	echo ========== End training RoDepParser ==========
	echo
}

if [[ $# -eq 0 ]]; then
	echo "Usage: train.sh [-split|-morph|-postag|-deppar|-all]"
	exit 1
fi

for MODULE in "$@"
do
	case $MODULE in
		-split)
			time train_sentence_splitter
			;;
		-morph)
			time train_morphology
			;;
		-postag)
			time train_pos_tagger
			;;
		-deppar)
			time train_dep_parser
			;;
		-all)
			time train_sentence_splitter
			time train_morphology
			time train_pos_tagger
			time train_dep_parser
			;;
	esac
done
