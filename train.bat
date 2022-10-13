@ECHO OFF

SETLOCAL ENABLEEXTENSIONS

IF [%1]==[-all] (
	CALL :Splitter
	CALL :Morpho
	CALL :Tagger
	CALL :Parser
	EXIT /B 0
)

IF [%1]==[-split] (
	CALL :Splitter
	EXIT /B 0
)

IF [%1]==[-morph] (
	CALL :Morpho
	EXIT /B 0
)

IF [%1]==[-postag] (
	CALL :Tagger
	EXIT /B 0
)

IF [%1]==[-deppar] (
	CALL :Parser
	EXIT /B 0
)

ECHO Usage: train.bat -all OR -split OR -morph OR -postag OR -deppar
EXIT /B 1

:Splitter
ECHO Training the sentence splitter...
:: Delete all model files for RoSentenceSplitter
DEL /F /Q data\models\splitter\model.pt
DEL /F /Q data\models\splitter_feat_len.txt
DEL /F /Q data\models\splitter_unic_props.txt
:: Retrain the sentence splitting model
python -m rodna.splitter
EXIT /B 0

:Morpho
ECHO Training the morphology...
:: Delete all model files for RoInflect
DEL /F /Q data\models\morphology\model.pt
DEL /F /Q data\models\char_ids.txt
DEL /F /Q data\models\unknown_aclasses.txt
:: Retrain the morphology model
python -m rodna.morphology
EXIT /B 0

:Tagger
ECHO Training the POS tagger...
:: Delete all model files for RoPOSTagger
DEL /F /Q data\models\tagger\cls\config.json
DEL /F /Q data\models\tagger\cls\model.pt
DEL /F /Q data\models\tagger\crf\config.json
DEL /F /Q data\models\tagger\crf\model.pt
DEL /F /Q data\models\tagger_unic_props.txt
DEL /F /Q data\models\word_ids.txt
:: Retrain the tagger
python -m rodna.tagger
EXIT /B 0

:Parser
ECHO Training the dependency parser
: Delete all model files for the RoDepParser models 1 and 2
DEL /F /Q data\models\parser\modelone.pt
DEL /F /Q data\models\parser\bert1\config.json
DEL /F /Q data\models\parser\bert1\pytorch_model.bin
DEL /F /Q data\models\parser\tok1\special_tokens_map.json
DEL /F /Q data\models\parser\tok1\tokenizer.json
DEL /F /Q data\models\parser\tok1\tokenizer_config.json
DEL /F /Q data\models\parser\tok1\vocab.txt
DEL /F /Q data\models\parser\modeltwo.pt
DEL /F /Q data\models\parser\bert2\config.json
DEL /F /Q data\models\parser\bert2\pytorch_model.bin
DEL /F /Q data\models\parser\tok2\special_tokens_map.json
DEL /F /Q data\models\parser\tok2\tokenizer.json
DEL /F /Q data\models\parser\tok2\tokenizer_config.json
DEL /F /Q data\models\parser\tok2\vocab.txt
: Retrain the parser model
python -m rodna.parser

ENDLOCAL
