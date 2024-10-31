@echo off
:: Set the code page to UTF-8
chcp 65001 >nul

set SCRIPT_PATH=.\result_auto_log.py
@REM set LABELS_TXT_PATH=..\TESTS\TEST9\labels.txt ^
@REM ..\TESTS\TEST9\labels.txt

@REM set SOURCE_IMG_DIR_PATH=D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試圖資\

@REM set WEIGHTS_PATH=d:\Documents\GitRepos\MaskRCNN_TF2\projects\TESTS\TEST9\output_weights\TEST9_300_60_100.h5 ^
@REM  d:\Documents\GitRepos\MaskRCNN_TF2\projects\TESTS\TEST9\output_weights\TEST9_300_80_100.h5

@REM set OUTPUT_PATH=D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試圖資\Test_Results\TEST9_300_60_100 ^
@REM  D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試圖資\Test_Results\TEST9_300_80_100

@REM set ANNOT_PATH=D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試圖資\annot

set LABELS_TXT_PATH=..\TESTS\TEST10\labels.txt

set SOURCE_IMG_DIR_PATH=C:\Users\Leo\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\測試圖資

set WEIGHTS_PATH=..\TESTS\TEST10\output_weights\mask_rcnn_custom_cfg_0185.h5

set OUTPUT_PATH=C:\Users\Leo\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\Test_Results\TEST10_e185

set ANNOT_PATH=C:\Users\Leo\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\測試圖資annot



setlocal EnableDelayedExpansion
:: Convert the lists into arrays
:: LABELS_TXT_PATH
set i=0
for %%D in (%LABELS_TXT_PATH%) do (
   set /A i+=1
   set "LABELS_TXT_PATH[!i!]=%%D"
)

:: WEIGHTS_PATH
set i=0
for %%D in (%WEIGHTS_PATH%) do (
   set /A i+=1
   set "WEIGHTS_PATH[!i!]=%%D"
)

:: OUTPUT_PATH
set i=0
for %%D in (%OUTPUT_PATH%) do (
   set /A i+=1
   set "OUTPUT_PATH[!i!]=%%D"
)



for /L %%j in (1,1,%i%) do (
    echo "python %SCRIPT_PATH% -l !LABELS_TXT_PATH[%%j]! -a %ANNOT_PATH% -w !WEIGHTS_PATH[%%j]! -d  %SOURCE_IMG_DIR_PATH% -o !OUTPUT_PATH[%%j]!" 
    python %SCRIPT_PATH% -l !LABELS_TXT_PATH[%%j]! -a %ANNOT_PATH% -w !WEIGHTS_PATH[%%j]! -d  %SOURCE_IMG_DIR_PATH% -o !OUTPUT_PATH[%%j]! 
)
endlocal

