---
layout: article
title: "Generating Instagram Style Captions and Hashtags"
categories: teampost
last_modified_at: 2020-03-02T13:00:00+09:00
tags: [cv]
external_teaser: "https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B77.png"
external_feature: "https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B77.png"
--- 

본 포스트에서는 사진이 주어졌을 때 마치 인스타그램에서 작성한 것과 같은 캡션과 해시태그를 생성하는 모델을 구현해보고 이를 서비스화 해본 내용을 담고 있습니다. 대표적인 Image Captioning 모델인 *Show, Attend and Tell* 을 기본으로 두고 여기에 몇 가지 다른 기법들을 추가하여 모델을 구현했습니다.

실험에 사용한 코드는 [DIYA 동아리 저장소](https://gitlab.diyaml.com/cv/2019-2-instagram-captioning)에서 확인하실 수 있습니다.

>학습은 [KC-ML2](https://www.kc-ml2.com)의 지원으로 AWS p2.2xlarge 인스턴스에서 진행하였습니다.


## Index
1. [Introduction](#intro)
2. [Experiments](#Experiments)
3. [Evaluation](#Evaluation)
4. [Result & Ablation Study](#Result)
5. [Web Demo](#Web_Demo)


## 1. <a name="intro">Introduction</a>

이미지 캡셔닝의 대표적인 두 논문을 우선 소개합니다. 첫번째로 소개해드릴 논문은 2014년도에 발표된 *Show and Tell* <sup>[1](#show)</sup>입니다. *Show and Tell* 은 기계번역의 원리에 착안하여 이미지를 input으로 받으면 이를 묘사하는 캡션을 output으로 생성하는 구조를 가집니다. 모델의 학습은 주어진 이미지에 대응되는 정확한 묘사의 확률을 최대화하는 방향으로 이루어집니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B71.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 1. 이미지 캡셔닝 maximum likelihood</font></figcaption>
</center>
<br>

아래 그림을 보면 *Show and Tell* 모델이 ent-to-end 구조로 되어 있음을 알 수 있습니다. 먼저 캡션이 생성될 이미지가 CNN 인코더를 통해 고정된 길이의 feature vector로 임베딩 됩니다. LSTM 디코더는 feature vector를 인풋으로 받아 문장을 생성합니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B72.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 2. Show and Tell의 구조</font></figcaption>
</center>
<br>

다음으로는 저희 프로젝트에서 baseline 모델로 활용한 *Show, Attend and Tell* <sup>[2](#attend)</sup>에 대해서 설명하겠습니다. 이 모델은 간단히 말하면 앞선 *Show and Tell* 의 디코더에 Attention mechanism을 추가한 모델입니다. 아래 그림을 보면 모델이 순차적으로 단어를 생성해낼 때에 이미지의 특징 중 어느 부분에 집중을 하였는지를 알 수 있습니다. 색이 하얗게 표시될수록 해당 부분의 영향이 크다는 것을 나타냅니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B73.png" style="width: 90%; height: 90%;"/>
<figcaption><font size=2pt>그림 3. Show Attend and Tell의 Attention 시각화 <sup>2</sup></font></figcaption>
</center>
<br>

*Show, Attend and Tell* 모델에서는 하나의 특징벡터만을 생성하는 *Show and Tell* 과 달리 CNN이 특징벡터 여러 개를 생성해냅니다. 디코더는 Attention mechanism을 통해 여러 특징벡터 중 어떤 특징벡터에 집중할지를 추가로 고려하여 캡션을 생성해냅니다. 논문에서는 인코더로 VGG를 사용하였으며, 마지막 convolutional layer로부터 특징벡터를 추출하였습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B74.png" style="width: 90%; height: 90%;"/>
<figcaption><font size=2pt>그림 4. CNN 인코더 예시</font></figcaption>
</center>
<br>

논문에서 소개하는 Attention mechanism에는 크게 두가지(Hard Attention, Soft Attention)가 있습니다. 저희 프로젝트에서는 back-propagation을 활용하여 End-to-end 방식으로 학습할 수 있는 Soft Attention 방식을 사용하였습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B75.png" style="width: 90%; height: 90%;"/>
<figcaption><font size=2pt>그림 5. Soft Attention 시각화 <sup>2</sup></font></figcaption>
</center>
<br>

아래 그림은 *Show, Attend and Tell* 에서 t번째 단어를 생성할 때 Attention Mechanism이 어떻게 작동하는지 보여줍니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B76.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>그림 6. Show, Attend and Tell에서 Attention을 사용한 캡션 생성</font></figcaption>
</center>
<br>

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B77.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>그림 7. Show, Attend and Tell - 적절한 캡션 생성 <sup>2</sup></font></figcaption>
</center>
<br>

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B78.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>그림 8. Show, Attend and Tell - 잘못된 캡션 생성 <sup>2</sup></font></figcaption>
</center>

## 2. <a name="Experiments">Experiments</a>

실험에 사용한 데이터셋은 동국대학교에서 공개한 Korean Tourist Spot Multi-Modal Dataset <sup>[3](#data)</sup>입니다. 해당 데이터셋은 해변(beach), 산(mountain), 놀이공원(amusement park) 등의 10개의 서브클래스로 이루어져 있고 각 서브클래스는 총 1000개의 사진 이미지, 캡션, 해시태그 셋으로 이뤄져 있습니다. 저희는 여기에 사람(person) 클래스를 추가하여 총 11,000개의 이미지에 대한 캡션, 해시태그 셋을 전체 데이터 셋으로 사용했습니다. 전체 11,000개의 데이터 중 7,700개는 학습 데이터셋, 2,200개는 validation 셋, 1100개는 테스트 셋으로 사용했습니다.

<br>
<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%91%E1%85%AD1.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>표 1. Korean Tourist Spot Multi-Modal Dataset class와 sub-class <sup>3</sup></font></figcaption>
</center>
<br>

*Show, Attend and Tell* 을 바탕으로 한 baseline 모델에서는 이미지의 feature를 추출하기 위한 인코더로 사전 학습된 ResNet-101을 사용했습니다. 디코더로는 위에서 소개한 Soft Attention을 사용하는 LSTM 아키텍처를 사용했습니다.

저희는 baseline 모델을 개선하기 위해 1) 인코더로 인스타그램 사진으로 사전 학습된 ResNeXt-WSL-32x8를 사용하였고, 2) 디코더에 Look-Back Attention을 추가했습니다.

인코더가 이미지의 feature를 잘 추출할수록 디코더는 보다 정확한 정보를 바탕으로 캡션을 생성할 수 있습니다. ResNeXt는 그림 9와 같이 데이터가 중간에 여러 path로 쪼개지고(split) 다시 합쳐지는(merge) 과정을 반복하는 구조를 가지고 있는 모델입니다. <sup>[4](#aggregated)</sup> 중간에 데이터가 나눠지는 path의 숫자를 Cardinality라고 하는데 이 값과 bottleneck을 적절히 조절하면 일반적인 ResNet과 동일한 파라미터의 개수를 유지할 수 있습니다. 하지만 모델의 크기를 ResNet과 동일하게 유지하더라도 아래에서 볼 수 있듯이 이미지를 분류하는 task에서 ResNeXt가 더 좋은 성능을 보이는 것을 알 수 있습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B79.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>그림 9. ResNet 구조 예(왼쪽)와 ResNeXt 구조 예(오른쪽) <sup>4</sup></font></figcaption>
</center>
<br>
<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B710.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>그림 10. ResNet과 ResNeXt, Top-1 error 비교 <sup>4</sup></font></figcaption>
</center>
<br>

저희가 사용한 ResNeXt-WSL은 ResNeXt 모델을 사용하여 인스타그램 9억장의 사진을 Weakly Supervised Learning 방법으로 학습한 모델로 논문에 따르면 object detection에서 매우 우수한 성능을 보입니다. 인스타그램 사진 데이터를 사용하여 학습 되었기 때문에 인스타그램 이미지 캡셔닝 task로의 transfer가 더 용이할 것으로 기대했습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B711.png" style="width: 50%; height: 50%;"/>
<figcaption><font size=2pt>그림 11. Look-Back Attention <sup>5</sup></font></figcaption>
</center>
<br>

Baseline 모델을 개선하기 위해 디코더에는 Look-Back Attention 메커니즘<sup>[5](#image)</sup>을 추가했습니다. Look-Back Attention은 이번 단계의 Attention을 생성할 때 이전 단계의 context vector를 추가적인 인풋으로 받습니다. 이것은 현재 단계에서 이미지의 어떤 부분을 봐야하는지를 정하기 위해 지난 단계에서는 어떤 부분을 집중적으로 봤는지 고려하는 방법입니다.

## 3. <a name="Evaluation">Evaluation</a>

이미지 캡셔닝의 품질을 측정하기 위해 BLEU-1, ROUGE-L, METEOR 세 metric을 사용했습니다.

[BLEU](https://en.wikipedia.org/wiki/BLEU)는 Bilingual Evaluation Understudy의 약자로 brevity penalty와 clipping을 사용하여 precision을 개선한 metric입니다. 저희가 사용한 BLEU-1은 unigram 단위만 고려하는 BLEU입니다.

[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))는 Recall Oriented Understudy for Gisting Evaluation의 약자로 BLEU가 precision을 보완한 metric이라면 ROUGE는 recall에 더 가중치를 주는 metric입니다. ROUGE는 precision과 recall의 가중조화평균으로 정의되는데, 이때 recall에 가중치를 많이 주게 됩니다. 저희가 사용한 ROUGE-L은 precision과 recall을 계산할 때 분자에 LCS (Longest Common Subsequence)가 들어가는 ROUGE의 한 종류입니다.

[METEOR](https://en.wikipedia.org/wiki/METEOR)는 Metric for Evaluation of Translation with Explicit Ordering의 약자로 ROUGE와 유사하게 recall에 가중치를 더 주는 recall과 precision의 가중조화평균입니다. 여기에 더해 METEOR는 그 이름이 암시하듯이 단어의 ordering에 따른 penalty를 추가했습니다.

해시태그 생성의 품질은 일반적인 precision과 recall 그리고 이 둘의 조화평균인 F1을 사용하여 측정했습니다.

## 4. <a name="Result">Result & Ablation Study</a>

### Result

Baseline 모델인 *Show, Attend and Tell* 의 아키텍처와 ResNeXt 인코더와 Look-Back Attention decoder를 사용한 저희 모델을 비교했습니다. 아래 표2에서 볼 수 있듯 모든 지표에서 저희 모델이 더 우수한 성능을 보이고 있습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%91%E1%85%AD2.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>표 2. Baseline 모델과 ResNeXt-Look-Back 모델 비교</font></figcaption>
</center>

### Ablation Study

저희는 추가적으로 ResNeXt 인코더와 Look-Back Attention 디코더가 각각 성능 향상에 얼마나 기여했는지 확인해보고자 Ablation study를 진행했습니다. 그 결과는 아래 표와 같습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%91%E1%85%AD3.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>표 3. baseline 모델과 Look-Back 모델 비교</font></figcaption>
</center>
<br>

위의 표에서 확인할 수 있듯이 Look-Back Attention만 추가한 모델의 경우 baseline 모델에 비해 약간의 성능 향상이 있었지만 ResNeXt 인코더와 Look-Back Attention을 모두 사용한 모델 (ours)에 비하면 성능이 떨어짐을 확인할 수 있습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-%E1%84%91%E1%85%AD4.png" style="width: 90%; height: 90%;"/>
<figcaption><font size=2pt>표 4. baseline 모델과 ResNeXt 모델 비교</font></figcaption>
</center>
<br>

ResNeXt 디코더만 추가한 모델의 경우, baseline 모델은 물론이고 ResNeXt encoder와 Look-Back Attention을 모두 사용한 모델 (ours)과 비교하여도 성능이 더 좋았습니다.

ResNeXt 인코더에 Look-Back Attention를 추가하였을 때 모델의 성능의 저하가 이루어진 이유를 다음과 같이 추측해 보았습니다.

* 캡션: 한국어에는 조사와 어미가 많이 등장하는데 ‘특정 단어 다음에 어떤 조사/어미를 사용할 것인가’는 이미지와는 거의 관련 없는, language modeling의 문제라고 판단됩니다. 때문에 조사와 어미를 생성하는 단계에서 이전 단계에서 이미지의 어떤 부분을 참고했는지를 고려하는 Look-Back Attention의 추가적인 정보가 유용하지 않습니다. 마찬가지로 조사와 어미 다음의 단어를 생성하는 단계에서도 조사와 어미는 이미지의 특정 부분에 의해 영향을 받지 않기 때문에  이전 단계에서 이미지의 어떤 부분을 참고했는지를 고려하는 Look-Back Attention의 추가적인 정보가 유용하지 않습니다.
* 해시태그: 텍스트와 다르게 하나의 해시태그와 그 다음 해시태그의 관계가 매끄럽게 연결되지 않습니다. 또한 해시태그의 경우 일반적으로 하나의 단어보다 더 많은 의미를 함축하고 있는 경우가 대부분입니다. 따라서 다음 단어를 생성할 때는 이전 단어와 함께 이전 단어를 생성할 때 이미지의 어떤 부분을 참고했는지에 대한 정보가 추가적으로 제공될 수 있지만 다음 해시태그를 생성할 때에는 이 추가적인 정보가 크게 유용하지 않았을 것으로 추측됩니다.

결론적으로 Look-Back Attention이 제공하는 추가적인 정보가 한국어 캡션과 해시태그 생성 유용하지 않고 오히려 아키텍쳐의 복잡도만 높였기 때문에 성능이 하락한 것으로 추측됩니다.

## 5. <a name="Web_Demo">Web Demo</a>

저희가 실험에 사용한 모델 중 가장 좋은 성능을 보였던 모델인 ResNeXt만을 추가한 모델을 활용하여 웹 데모를 만들었습니다. 웹 데모 소개로 포스팅을 마치겠습니다.

#### [http://cv.diyaml.com](http://cv.diyaml.com)

아래는 데모 페이지의 예시입니다. 사진을 업로드하면 인스타그램 형식으로 캡션과 해시테그가 생성되는 것을 보실 수 있습니다. Web Application으로는 Flask를 활용했습니다.

<center>
<img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019CV/2-demo.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 12. 데모 페이지 예시</font></figcaption>
</center>


## References

##### <sup><a name="show"></a>1</sup><a href="https://arxiv.org/pdf/1411.4555.pdf" target="_blank"><sub>Show and Tell: A Neural Image Caption Generator, Vinyals et al, arXiv:1411.4555</sub></a>

##### <sup><a name="attend"></a>2</sup><a href="https://arxiv.org/pdf/1502.03044.pdf" target="_blank"><sub>Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, Xu et al, arXiv:1502.03044</sub></a>

##### <sup><a name="data"></a>3</sup><a href="https://pdfs.semanticscholar.org/eabf/579142dbd736f8cc6bb0ebf3873ec0fd502e.pdf?_ga=2.8003378.1859059531.1612360690-696917962.1612360690" target="_blank"><sub>Korean Tourist Spot Multi-Modal Dataset for Deep Learning Applications, Jeong et al</sub></a>

##### <sup><a name="aggregated"></a>4</sup><a href="https://arxiv.org/pdf/1611.05431.pdf" target="_blank"><sub>Aggregated Residual Transformations for Deep Neural Networks, Xie et al, arXiv:1611.05431</sub></a>

##### <sup><a name="image"></a>5</sup><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_Look_Back_and_Predict_Forward_in_Image_Captioning_CVPR_2019_paper.pdf" target="_blank"><sub>Look back and predict forward in image captioning, Qin et al, CVPR2019</sub></a>

-----------

 Written by 
  문지환 | 박수현 | 윤준석 | 주윤하
