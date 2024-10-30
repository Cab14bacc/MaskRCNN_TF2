@echo off
:: Set the code page to UTF-8
chcp 65001 >nul

set SCRIPT_PATH=.\convert_VGG_annot_to_custom_annot.py 
set SOURCE_IMG_DIR=D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\測試圖資
set OUTPUT_PATH=D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\測試圖資annot

:: an array of subfolders of the output path 
set SUBFOLDERS=Google 世曦 詹老師jpg 詹老師tif

for %%F in (%SUBFOLDERS%) do (
    echo Processing %%F
    python  %SCRIPT_PATH% -o %OUTPUT_PATH%\%%F\converted_annot.json -s %OUTPUT_PATH%\%%F\annot.json -d %SOURCE_IMG_DIR%\%%F

)