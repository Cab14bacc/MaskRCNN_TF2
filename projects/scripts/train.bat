@echo off

set SCRIPT_PATH=.\custom_training.py
set TEST_PATH=..\TESTS\TEST10
set WEIGHTS_PATH=..\mask_rcnn_coco.h5


python %SCRIPT_PATH% -s %TEST_PATH% -w %WEIGHTS_PATH% -he 10 -ae 300
