#!/bin/bash

function train_sentence_splitter {
	echo
	echo ========== Training RoSentenceSplitter ==========
	echo

	# Delete all model files for the RoSentenceSplitter model
	rm -fvr data/models/splitter/*
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
	rm -fvr data/models/morphology/*
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
	rm -fvr data/models/tagger/cls/*
	rm -fvr data/models/tagger/crf/*
	rm -fv data/models/tagger_unic_props.txt
	rm -fv data/models/word_ids.txt
	# Retrain the tagger model
	python3 -m rodna.tagger

	echo
	echo ========== End training RoPOSTagger ==========
	echo
}

if [[ $# -eq 0 ]]; then
	echo "Usage: train-all.sh [-split|-morph|-postag|-all]"
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
		-all)
			time train_sentence_splitter
			time train_morphology
			time train_pos_tagger
			;;
	esac
done
