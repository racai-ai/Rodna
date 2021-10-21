@ECHO OFF

SETLOCAL ENABLEEXTENSIONS

REM Delete all model files for the RoPOSTagger LM model
RMDIR /S /Q data\models\tagger\langmod\assets
RMDIR /S /Q data\models\tagger\langmod\variables
DEL /F /Q data\models\tagger\langmod\*.pb
DEL /F /Q data\models\tagger_unic_props.txt
DEL /F /Q data\models\word_ids.txt
REM Retrain the LM model
python -m rodna.tagger -lm

ENDLOCAL
