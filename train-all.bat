@ECHO OFF

SETLOCAL ENABLEEXTENSIONS

:: REM Delete all model files for RoInflect
:: RMDIR /S /Q data\models\morphology\assets
:: RMDIR /S /Q data\models\morphology\variables
:: DEL /F /Q data\models\morphology\*.pb
:: DEL /F /Q data\models\char_ids.txt
:: DEL /F /Q data\models\unknown_aclasses.txt
:: REM Retrain the morphology model
:: python -m rodna.morphology

REM Delete all model files for RoPOSTagger
RMDIR /S /Q data\models\tagger\assets
RMDIR /S /Q data\models\tagger\variables
DEL /F /Q data\models\tagger\*.pb
DEL /F /Q data\models\tagger_unic_props.txt
DEL /F /Q data\models\word_ids.txt
REM Retrain the LM model
python -m rodna.tagger

ENDLOCAL
