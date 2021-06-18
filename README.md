# Capstone design 1 (1)
## X-Ray Convolutional Neural Network with NIH dataset
> NIH Dataset으로의 학습을 통해 향후 Cardiomegaly(심장비대증) classification 전이학습에 weight를 사용한다.



### Dataset

112,120개의 14개의 병변이 multi label로 구성된 ChestX-ray dataset을 사용하였다.   
0 : Normal xray  
1 : Abnormal xray without Cardiomegaly  
2 : Abnormal xray with Cardiomegaly



### Preprocessing
전처리에 사용된 기법은 아래와 같다.
  * Contrast Limited Adaptive Histogram Equalization (CLAHE) to correct contrast(might introduce some error)
  * Resize images from 1024x1024p to 256x256p  
  (resize되는 값이 크면 클수록 좋다)

<!---
[//]: # (![Xray after applying contrast](https://i.imgur.com/Z9aIY77.png))
-->




### Usage
  * ```git clone```
  * ChestX-ray14 database [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)에서 데이터셋 다운로드
  * 개별 디렉토리에 압축을 푼다 (e.g. images_01.tar.gz into images_01).
  * ```python PreprocessData.py```
  * ```python Main.py```