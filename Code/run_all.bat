@echo off
echo Running all tests...

:: Chest + AlexNet
python train.py --data chest --model AlexNet --channels 1 --num_classes 2 --epochs 5

:: Organs + ResNet18
python train.py --data organs --model ResNet18 --channels 1 --num_classes 11 --epochs 8

:: Orgs + AlexNet
python train.py --data orgs --model AlexNet --channels 1 --num_classes 11 --epochs 10

:: Lesions + AlexNet
python train.py --data lesions --model AlexNet --channels 3 --num_classes 7 --epochs 8

:: Cells + AlexNet
python train.py --data cells --model AlexNet --channels 3 --num_classes 8 --epochs 8

echo All tests complete!
pause