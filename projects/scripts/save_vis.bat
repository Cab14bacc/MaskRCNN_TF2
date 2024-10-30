@echo off
:: Set the code page to UTF-8
chcp 65001 >nul

set SCRIPT_PATH=.\custom_prediction_save_visualization.py 
set LABELS_TXT_PATH=..\TESTS\TEST6\labels.txt 
set SOURCE_IMG_DIR_PATH=D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試圖資\
set WEIGHTS_PATH=d:\Documents\GitRepos\MaskRCNN_TF2\projects\TESTS\TEST6\output_weights\TEST6_300_180_2024-09-16-03-57.h5
set OUTPUT_PATH=D:\Documents\ComputerGraphics\RoadMarkingsDectection\Test_Results\TEST6_300_180_2024-09-16-03-57

:: an array of subfolders of the output path 
set SUBFOLDERS=Google 世曦 詹老師jpg 詹老師tif

for %%F in (%SUBFOLDERS%) do (
    echo Processing %%F
    python  %SCRIPT_PATH% -l  %LABELS_TXT_PATH% -w  %WEIGHTS_PATH% -s %SOURCE_IMG_DIR_PATH%\%%F -o %OUTPUT_PATH%\%%F
)

