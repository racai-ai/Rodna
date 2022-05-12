@ECHO OFF

SETLOCAL ENABLEEXTENSIONS

:: Delete all model files for RoSentenceSplitter
DEL /F /Q data\models\splitter\model.pt
DEL /F /Q data\models\splitter_feat_len.txt
DEL /F /Q data\models\splitter_unic_props.txt
:: Retrain the sentence splitting model
python -m rodna.splitter

:: Delete all model files for RoInflect
DEL /F /Q data\models\morphology\model.pt
DEL /F /Q data\models\char_ids.txt
DEL /F /Q data\models\unknown_aclasses.txt
:: Retrain the morphology model
python -m rodna.morphology

:: Delete all model files for RoPOSTagger
:: RMDIR /S /Q data\models\tagger\cls\assets
:: RMDIR /S /Q data\models\tagger\cls\variables
:: DEL /F /Q data\models\tagger\cls\*.pb
:: RMDIR /S /Q data\models\tagger\crf\assets
:: RMDIR /S /Q data\models\tagger\crf\variables
:: DEL /F /Q data\models\tagger\crf\*.pb
:: DEL /F /Q data\models\tagger_unic_props.txt
:: DEL /F /Q data\models\word_ids.txt
:: Retrain the LM model
:: python -m rodna.tagger

ENDLOCAL
