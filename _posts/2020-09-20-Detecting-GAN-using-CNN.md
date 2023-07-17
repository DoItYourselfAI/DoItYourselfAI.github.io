---
layout: article
title: "Detecting GAN using CNN"
categories: teampost
last_modified_at: 2020-09-20T13:00:00+09:00
tags: [GAN, CNN]
external_teaser: "https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/msgstylegan/msgstylegan.png" 
external_feature: "https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/msgstylegan/msgstylegan.png" 
excerpt: "본 포스트는 실제 사람 이미지와 GAN을 통해 생성된 사람 이미지를 CNN을 통해 구분하여 성능을 측정해보고, 어떤 부분을 보고 구분을 하는지에 대한 내용을 담고 있습니다."
---

본 포스트는 실제 사람 이미지와 GAN을 통해 생성된 사람 이미지를 CNN을 통해 구분하여 성능을 측정해보고, 어떤 부분을 보고 구분을 하는지에 대한 내용을 담고 있습니다.

## Contents

1. [Introduction](#intro)
2. [Experiments](#experiments)
3. [Analysis & Conclusion](#concl)


## <a name="intro">1. Introduction</a>

딥페이크(Deepfake)란 딥러닝(deep learning)과 가짜(fake)의 합성어로, 인공지능 기술을 이용하여 영상, 이미지 속 인물의 얼굴이나 음성 등을 다른 사람의 것으로 바꾸는 기술을 말합니다. 딥페이크 이미지 합성 방법으로 4가지가 있으며, 모든 방법에 GAN을 적용하여 성능을 높일 수 있습니다.<sup>1</sup>

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/manipulation_4.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt> 그림 1. 다양한 이미지 합성 방법 <sup>1</sup></font></figcaption>
</center>

첫번째 이미지 합성 방법인 'Entire Face Synthesis'는 실제 존재하지 않는 인물의 얼굴을 이미지나 영상으로 생성하는 방법입니다. 참고하는 이미지 없이 이미지를 생성하는 방법으로, 대표적으로 GAN이 있습니다.

두번째 이미지 합성 방법인 'Identity Swap'는 이미지나 영상 속 인물의 얼굴을 타인의 얼굴로 바꾸는 방법입니다. 유명인의 얼굴을 합성한 성인물을 인터넷에서 유포하는 데 악용되고 있습니다.

세번째 이미지 합성 방법인 'Attribute Manipulation'는 이미지나 영상 속 인물의 얼굴 일부(머리카락, 안경, 눈 등)를 바꾸는 방법입니다. SNOW, FaceApp 등의 사진 앱에서 볼 수 있듯이, 상용화가 잘 되어있습니다.

마지막 이미지 합성 방법인 'Expression Swap'는 이미지나 영상 속 인물의 표정(facial expression)을 타인의 표정으로 바꾸는 방법입니다. 단순히 감정 표현뿐만 아니라 입모양, 눈 깜빡임 등도 바꿀 수 있기 때문에 유명인의 연설 혹은 발표 장면에 의도된 입모양을 합성 후 위조된 음성을 추가하여 가짜뉴스를 생성, 배포하는 데 악용되고 있습니다.

이러한 4가지 이미지 합성 방법들 중에서 저희 팀은 첫번째 소개한 'Entire Face Synthesis' 방법으로 생성된 이미지와 실제 이미지를 구분하는 문제를 주제로 잡았습니다. 위 문제는 개, 고양이 분류와 같은 방법으로 문제를 해결할 수 있을 것이라 판단했습니다. 실제로 manipulation detection을 다룬 기존 연구들<sup>2,3,4</sup>에서도 CNN을 통해 위 문제를 해결하였으며 GAN 네트워크가 이미지를 생성할 때 고유의 fingerprint로 인해 이를 구분한다는 것입니다. GAN fingerprint는 사람의 눈으로는 판별할 수 없는 미세한 패턴이지만 딥러닝 모델을 통해서는 학습이 가능한 특징이기 때문에 CNN을 통해 위 문제를 해결해보고자 했습니다.


## <a name="experiments">2. Experiments</a>

실험 코드는 [동아리 gitlab repo](https://gitlab.diyaml.com/cv2020/realfake)를 통해 확인할 수 있습니다.

### 2.1. 실험 설계

Classification 기반으로 GAN detection이 가능할 것이라는 가정하에 우리는 2가지 방식의 실험을 진행하였습니다. 첫 번째는 binary classification 실험으로, 진짜 이미지와 여러 GAN 모델을 통해 생성된 합성 이미지들을 통해 모델이 'real'과 'fake'를 판별해내도록 모델을 학습시켰습니다. 두 번째는 multi-class classification으로, 4가지 GAN 모델을 통해 생성된 가짜 이미지들을 통해서 주어진 이미지가 어떠한 GAN을 통해 생성이 되었는지 분류해내도록 모델을 학습하였습니다.

### 2.2. 데이터셋

저희는 GAN 연구에서 흔히 사용되는 CelebA-HQ(1024 x 1024)와 FFHQ(1024 x 1024)를 통해 학습된 4가지 GAN 네트워크를 통해 가짜 이미지들을 생성했습니다. MSG-GAN<sup>2</sup>, StyleGAN<sup>3</sup>, PGGAN<sup>4</sup>, VGAN<sup>5</sup>을 사용하였고, 각각의 GAN 모델들에서 1만 장의 이미지를 생성하여 총 4만장의 데이터셋을 구성하였습니다. Multi-class classification에서는 CelebA-HQ 데이터셋으로 학습된 GAN 모델들로 생성한 이미지를 사용하였고, Real-fake binary classification에서는 FFHQ 이미지 원본을 real label 데이터로, FFHQ로 학습된 GAN 모델들이 생성한 이미지를 fake label 데이터로 사용했습니다.

### 2.3. 실험 환경

실험 환경은 다음과 같습니다.
- CPU : 18-core Intel Xeon E5-2695 @ 2.10 GHz
- GPU : 2 NVIDIA Titan X
- OS : Ubuntu 18.04, CUDA 10.1, cuDNN 7, Python 3.7.7
- Framework
  * Image Generation : TF v1.15.0
  * GAN Detection : PyTorch v1.6.0

### 2.4. 실험 결과

#### 2.4.1. Binary Classification

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/resnet50_binary_training_log.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt> 그림 2. ResNet-50의 Training curve </font></figcaption>
</center>

'real'과 'fake'를 판별하는 첫 번째 실험에서는 ResNet-50를 baseline network로 사용했습니다. ImageNet에서 pretrained한 네트워크에서 linear classifier를 1000에서 2로 수정한 후, fine-tuning을 진행하였습니다. 그 결과 위의 그림과 같이 99%의 정확도를 보였습니다.

#### 2.4.2. Multi-Class Classification

앞에서 'real'과 'fake'를 판별하는 것은 쉽게 해결될 수 있음을 확인하였습니다. 그렇다면 task의 난이도를 높여서 주어진 합성 이미지가 어떤 GAN을 통해 만들어진 것인지 판별하는 것도 가능할 지에 대한 실험도 진행하였습니다. Binary classification에서와 같이 ResNet-50를 baseline network로 사용하였으며 ImageNet에서 pretrained한 네트워크에서 linear classifier를 1000에서 4로 수정한 후, fine-tuning을 진행하였습니다. 이번에는 baseline network 이외에 ResNet-101, Xception network를 추가하여 실험을 진행했습니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/resnet101_training_log.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt> 그림 3. ResNet-101의 Training curve </font></figcaption>
</center>

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/xception_training_log.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt> 그림 3. Xception의 Training curve </font></figcaption>
</center>

실험 결과 ResNet과 Xception 모두 binary classification에서와 마찬가지로 99%의 정확도에 도달했습니다. 'real', 'fake'를 판별하는 것 뿐 아니라 주어진 이미지가 MSG-GAN, StyleGAN, PGGAN, VGAN 중 어떤 모델에서 생성된 것인지 구분하는 것 역시 완벽에 가깝게 해낼 수 있었습니다.


## <a name="concl">3. Analysis & Conclusion</a>

Detection model이 이미지의 어떤 부분을 보고 구분하는 지를 확인하기 위해 Grad-CAM을 생성했습니다. 아래의 예시들은 특정 GAN에서 생성된 이미지가 입력으로 주어졌을 때 Xception을 baseline으로 한 GAN detection 모델의 12번째 블록에서 뽑은 Grad-CAM의 결과들입니다. Grad-CAM에서 색칠된 부분은 모델이 해당 클래스로 이미지를 분류할 때 가장 큰 비중을 둔 곳으로 파란색에서 빨간색으로 갈수록 비중의 값이 커집니다.

**1) MSG-GAN**

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/msgstylegan/msgstylegan.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt> 그림 4. Grad-CAM on MSG-GAN </font></figcaption>
</center>

**2) StyleGAN**

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/stylegan/stylegan.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt> 그림 5. Grad-CAM on StyleGAN </font></figcaption>
</center>

**3) PGGAN**

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/pggan/pggan.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt> 그림 6. Grad-CAM on PGGAN </font></figcaption>
</center>

**4) VGAN**

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020CV/images/vgan/vgan.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt> 그림 7. Grad-CAM on VGAN </font></figcaption>
</center>

위의 예시들에서 볼 수 있듯이, 각각의 GAN마다 Grad-CAM이 찍히는 부분의 패턴이 존재합니다. 실제로 여러 이미지들에 대해 Grad-CAM을 생성한 결과 예시와 같은 패턴이 규칙적으로 발생함을 확인할 수 있었으며, 이를 통해 Manipulation detection 시에 detection 모델은 이미지 상에서 어색한 신체 부분(e.g. 눈, 코, 입, 귀)을 통해 진위를 판별하거나 GAN의 종류를 구분해 내는 것이 아니라, GAN의 아키텍처 특징에 따라서 발생하는 노이즈 패턴을 포착하여 판단을 함을 알 수 있었습니다.

하지만 이러한 모델에는 단점이 있는데, 바로 한번도 보지 못한 생성된 이미지에 대하여 구분을 못한다는 것입니다. 이러한 단점들은 현재 아직까지도 완전히 극복하지 못했기에 충분히 재미있는 연구가 되리라 생각합니다.


## References

##### <sup>1</sup><sub>[DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/pdf/2001.00179.pdf)</sub>

##### <sup>2</sup><sub>[Do GANs Leave Artificial Fingerprints?](https://arxiv.org/pdf/1812.11842.pdf)</sub>

##### <sup>3</sup><sub>[Source Generator Attribution via Inversion](https://arxiv.org/pdf/1905.02259.pdf)</sub>

##### <sup>4</sup><sub>[On the Detection of Digital Face Manipulation](https://arxiv.org/pdf/1910.01717.pdf)</sub>

##### <sup>5</sup><sub>[MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks](https://arxiv.org/pdf/1903.06048.pdf)</sub>

##### <sup>6</sup><sub>[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)</sub>
 
##### <sup>7</sup><sub>[Progressive Growing of GANS for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)</sub>

##### <sup>8</sup><sub>[Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/pdf/1810.00821.pdf)</sub>

-----------

 Written by 
  구윤모 | 노우준 | 이재성 | 주윤하
