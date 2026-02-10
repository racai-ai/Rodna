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
python -m processor.splitter
EXIT /B 0

:Morpho
ECHO Training the morphology...
:: Delete all model files for RoInflect
DEL /F /Q data\models\morphology\model.pt
DEL /F /Q data\models\char_ids.txt
DEL /F /Q data\models\unknown_aclasses.txt
:: Retrain the morphology model
python -m processor.morphology
EXIT /B 0

:Tagger
ECHO Training the POS tagger...
:: Delete all model files for RoPOSTagger
DEL /F /Q data\models\tagger\cls\config.json
DEL /F /Q data\models\tagger\cls\model.pt
DEL /F /Q data\models\tagger\crf\config.json
DEL /F /Q data\models\tagger\crf\model.pt
DEL /F /Q data\models\tagger\cls_bert\*.json
DEL /F /Q data\models\tagger\cls_bert\model.safetensors
DEL /F /Q data\models\tagger\cls_bert\vocab.txt
DEL /F /Q data\models\tagger\crf_bert\*.json
DEL /F /Q data\models\tagger\crf_bert\model.safetensors
DEL /F /Q data\models\tagger\crf_bert\vocab.txt
DEL /F /Q data\models\tagger_unic_props.txt
:: Retrain the tagger
python -m processor.tagger
EXIT /B 0

:Parser
ECHO Training the dependency parser
:: Model files for the RoDepParser models 1 and 2 are automatically deleted now
:: Retrain the parser model
python -m processor.parser

ENDLOCAL
