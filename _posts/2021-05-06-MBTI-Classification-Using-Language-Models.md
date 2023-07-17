---
layout: article
title: "MBTI Classification Using Language Models"
categories: teampost
last_modified_at: 2021-05-06T02:00:00+09:00
tags: [NLP]
external_teaser: "https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/MARVEL MBTI.png" 
external_feature: "https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/MARVEL MBTI.png" 
excerpt: "이 포스트는 Kaggle의 MBTI 데이터셋을 분석해 각 16개의 타입으로 분류하는 모델을 구현해본 내용을 담고 있습니다."
---

이 포스트는 [Stanford의 CS224n](http://web.stanford.edu/class/cs224n) 강의에서 배운 내용을 바탕으로 [Kaggle의 MBTI 데이터셋](https://www.kaggle.com/datasnaek/mbti-type)을 분석해 각 16개의 타입으로 분류하는 모델을 구현해본 내용을 담고 있습니다. Count-based Vectorization을 이용한 고전적인 머신러닝 알고리즘과 Language Model을 이용한 딥러닝 알고리즘을 구현하여 각각의 장단점을 비교해보고자 하였습니다.

모델 구현에 사용한 코드는 [DIYA 동아리 저장소](https://gitlab.diyaml.com/nlp2020/mbti-kaggle)에서 확인하실 수 있습니다.
>DIYA는 동아리 활동 및 GitLab 운영에 머신러닝 연구조직 [KC-ML2](https://www.kc-ml2.com)의 지원을 받고 있습니다.  

## 목차

1. [Why MBTI?](#intro)
2. [Dataset & Our Method](#dataset)
3. [Experimental Results](#experiments)


## 1. <a name="intro">Why MBTI?</a>

### 1.1. MBTI란 무엇인가요?

Myers-Briggs Type Indicator (이하 MBTI)는 스위스의 정신의학자 칼 구스타브 융이 분석한 성격 유형 이론에 근거해 이사벨 마이어스와 캐서린 쿡 브릭스가 만든 성격 유형 지표입니다. 사람이 특정한 정보를 인식하고 또 판단하는 방식에 차이가 있다는 관찰로부터 출발했으며, 사람 간의 성격적 차이와 이로 인한 갈등에 대해 이해하기 위해 만들어졌습니다.

에너지의 방향, 인식 기능, 판단 기능, 이해/생활 양식의 네 가지 분류에 대해 양자택일의 형식으로 검사를 진행합니다. 이는 인간의 영혼이 대립적인 요소로 이루어져 있지만 결국 조화를 이룬다는 칼 융의 대극이론에 기반합니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/8_MBTI_Letters.jpg" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 1. MBTI의 4가지 대립적 요소</font></figcaption>
</center>

이렇듯 이분법적인 카테고리 네 가지가 합쳐져 총 16가지의 성격유형을 구분하게 됩니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/16 MBTI Types.jpg" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 2. 16가지 MBTI 유형 </font></figcaption>
</center>

현재  Myers-Briggs Company가  1975년부터  검사를  출판  및  시행하고  있으며  개인은  약 50달러에  검사를  진행할 수 있습니다. 우리나라의 경우 MBTI의 라이선스를 소지하고 있는 [심리평가기관](https://www.career4u.net)을 통해 유료로 온라인 검사를 받아보실 수 있습니다.

### 1.2. 왜 MBTI 데이터를 선택했나요?

MBTI 검사는 본인이 생각하는 자신의 모습을 기준으로 설문을 작성하는 자기보고식 검사이기에 주어진 상황이나 조건에 따라 결과가 다르게 나올 수 있습니다. 따라서 MBTI 성격 유형과 다른 사람들이 보는 개인의 실제 성격에는 괴리가 있을 수 있습니다. 그럼에도 저희가 MBTI를 선택한 이유는 다음과 같습니다.

가장 먼저 MBTI는 매년 2백만명의 사람이 검사하고 있을만큼 인기가 있습니다. 한국에서는 코로나 이후 MBTI를 카피한 [www.16personalities.com](http://www.16personalities.com) 이라는 무료 성격 유형 검사가 유행했습니다. 유튜브, 페이스북 등의 SNS로 빠르게 퍼져나가며 각종 커뮤니티에 상황극이나 밈(meme)이 인기를 끌었습니다. 이제는 인터넷에서 쉽게 유명인들의 MBTI 성격 유형을 찾아볼 수 있으며 MBTI를 비롯한 다양한 성격 유행 테스트들이 대중화되었습니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/MARVEL MBTI.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 3. 마블 히어로들의 MBTI 유형</font></figcaption>
</center>

빅데이터를 통해 개인화된 광고와 콘텐츠를 보여주는 시대입니다. 업로드한 글을 통해 쉽고 효율적으로 개인의 성격을 파악할 수 있다면 기업에도 이익이 될 것입니다. 기업이 새로운 고객집단을 확보하는 데는 기존 고객집단을 유지하는 것보다 다섯배의 비용이 더 든다고 합니다.<sup>1</sup> 온라인 고객 관계 관리의 (electronic customer relationship management) 측면에서 불특정 다수의 고객집단에 어필하는 것보다 좀 더 이익이 되는 고객집단을 대상으로 관계 유지 및 개선에 투자한다면 이익을 더욱 극대화할 수 있습니다.

따라서 알고리즘을 이용해 간단하게 소비자의 심리를 분석하는 방법이 발달한다면 소비자가 원하는 상품을 기획 및 제작하는 것 뿐 아니라 기업이 이익을 추구하는 방향에서 도움이 될 것입니다. 저희는 추후 이러한 알고리즘 개발을 염두에 두고, 이를 위한 개념 증명으로써 (Proof of Concept) MBTI 데이터를 먼저 분석해보기로 하였습니다.

## 2. <a name="dataset">Dataset & Our Method</a>

저희는 MBTI 데이터를 통해 [자연어 처리의 4가지 단계](https://blog.diyaml.com/teampost/%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC%EC%9D%98-4%EA%B0%80%EC%A7%80-%EB%8B%A8%EA%B3%84/)가 각각 성능에 얼마나 큰 영향을 미치는지 실험해보고자 했습니다. 데이터의 원자료는 Kaggle에 업로드되어 있는 [mbti-type 데이터셋](https://www.kaggle.com/datasnaek/mbti-type)을 사용했습니다. 해당 데이터셋은 총 8,675명의 사람들이 각각 적은 포스트 50개씩, 즉 총 433,750개의 포스트에 대해 작성자의 MBTI 성격 유형 라벨이 부여되어 있는 자료입니다.

### 2.1. Preprocessing

포스트들을 웹페이지에서 그대로 크롤링해서 얻은 데이터인만큼, 바로 토큰화를 진행하기는 어려워 아래와 같은 전처리 작업을 먼저 해주었습니다.

1. URL과 같은 hypertext를 제거했습니다.
2. Áóé 와 같은 accented character들을 제거했습니다. 
3. 축약어들을 원래의 형태대로 풀어서 다시 표기해주었습니다. (e.g. don't -> do not)
4. 모든 단어들을 원형화(lemma)했습니다. 여기서 원형화란 sleep, sleeping 등과 같이 기반이 되는 뜻은 동일하나 어미, 접사 등의 결합으로 인해 형태가 달라진 용언들을 원형(sleep)으로 바꾸어주는 일을 말합니다.
5. 불용어(stopwords)를 제거했습니다. 불용어란 for, to, the와 같이 등장하는 빈도수는 높지만 글의 내용이나 문맥과 관련이 없는 단어들을 이야기합니다.

    그런데 전처리를 하다보니 많은 포스트에서 스스로의 MBTI 유형을 직접 언급하고 있다는 것을 알게 되었습니다. 아래는 이해를 돕기 위해 원문으로부터 살짝 변형한 예시들입니다.

    > Hi, I am entj ...

    > ... as an ixfp, I believe ...

    첫번째 예시에서처럼 본인의 MBTI를 정확하게 명시하는 경우도 있었고, MBTI의 일부를 x 표시로 대체하여 보다 넓은 카테고리를 이용해 본인의 MBTI를 지칭하는 경우도 있었습니다. 또한 자신이 아닌 다른 MBTI 유형에 대해 언급하는 경우도 많았습니다.

    이렇게 본문에서 레이블이 직접적으로 노출되는 경우 머신러닝 및 딥러닝 알고리즘의 성능을 올바르게 평가하기 어려울 것이라 생각해, 저희는 첫번째 예시에서처럼 MBTI 유형의 4글자가 전부 적혀있는 경우 이를 \<MBTI>라는 특수한 토큰으로 대체해 분석을 진행하였습니다. 만일 MBTI 유형을 그대로 자연어 처리에 활용할 경우, 알고리즘이 문맥을 이해해서 MBTI 유형을 맞췄는지, 아니면 본문에 적혀있는 MBTI 유형을 그대로 출력했을 뿐인지 알 수 없기 때문입니다. 한편 두번째 예시와 같이 MBTI 유형이 일부 가려진 경우 그래도 가려진 카테고리에 대해서는 문맥을 통해 유추해야 하므로 남겨두기로 결정했습니다.

6. MBTI 유형을 포스트 본문에 명시한 경우 이를 \<MBTI>라는 토큰으로 대체했습니다.

    또 한가지 특기할 만한 점으로는 1번 전처리 과정에서 제거한 대부분의 URL이 유튜브 링크(95%)라는 점이 있었습니다. 원자료를 수집한 곳이 친목을 위한 웹페이지인 만큼 자신이 좋아하는 음악, 영상 등을 자신과 동일한 MBTI 유형의 사람들에게 추천하는 글이 많았습니다. 저희는 이러한 음악, 영상 등의 내용을 자연어 처리 과정에 반영한다면 알고리즘의 성능이 향상되지 않을까 기대하며 유튜브 링크를 통해 해당 영상의 제목을 찾고, URL이 있던 자리에 제목을 넣은 뒤 \<hypertext>라는 특수한 토큰으로 감싸주었습니다.

    이후 이미지나 gif가 삽입되어있는 경우에도 해당 hypertext가 있던 자리에 alt tag의 내용물을 대신 넣어주고 마찬가지로 \<hypertext>로 감싸주었습니다. alt tag는 웹페이지를 렌더링할 때 이미지가 불러와지지 않는 경우 대신 표시하는 텍스트를 말합니다. 예를 들어 나비 사진이 있으면 그 이미지 파일을 삽입하는 html tag 중 alt tag 안에 (일반적으로) "나비"라고 적혀있습니다.

7. 이미지, 영상 링크 등의 hypertext가 있던 자리에 alt tag의 내용물이나 영상 제목을 넣어주었습니다.

6번과 7번이 자연어 처리 성능에 얼마나 영향을 주는지 알아보기 위해 우선 1번부터 5번까지만 전처리를 수행해보고, 1번부터 6번까지, 그리고 1번부터 7번까지 전부 수행해보았습니다. [실험 결과](#experiments)에서는 각각의 전처리 방법을 다음과 같이 표기하였습니다.

Preprocessing | 전처리 내용
-- | --
Original | 1번(hypertext 제거)부터 5번(불용어 제거)까지만 수행
Masked | 6번(MBTI 대체)까지 추가로 수행
Hypertext | 7번(hypertext 내용 추가)까지 모두 수행


### 2.2. Embedding

머신러닝, 딥러닝 분류 알고리즘을 활용하기 앞서 유의미한 임베딩을 만드는 일이 얼마나 성능에 영향을 미치는지를 알아보기 위해 저희는 one-hot encoding만 사용하는 경우와 언어모델을 사용하는 경우를 비교해보았습니다.

One-hot encoding을 사용하는 경우에는 하나의 포스트를 그 안에 들어있는 모든 벡터들의 합으로 나타냈습니다. 즉 각 토큰의 index에 맞춰 해당 토큰이 포스트 안에 몇 번 등장하였는지를 세서 그 빈도를 나타낸 것인데요, 이렇게 토큰들의 등장 빈도로 문장, 문서의 임베딩을 구하는 방법을 Bag-of-Words(BoW)라고 부릅니다. python의 경우 scikit-learn <sup>2</sup>패키지의 [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)를 이용하면 아래와 같이 간단하게 BoW를 구현할 수 있습니다.

```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.toarray())
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
```

이때 `CountVectorizer`의 parameter로 `max_df`와 `min_df`를 명시해줄 수 있는데요, df란 document frequency의 약자로서 이 parameter들을 이용하면 각 포스트에 지나치게 많이 등장하거나 적게 등장하는 토큰들은 BoW에서 제외할 수 있습니다. 지나치게 많이 등장하는 토큰들을 제거하면 um, well과 같이 큰 의미는 없지만 불용어 리스트에는 미처 넣지 못한 토큰들을 제거할 수 있고, 적게 등장하는 토큰들을 제거하면 고유명사나 의미를 알 수 없는 오타 등을 제거할 수 있습니다. 저희는 이어서 서술할 머신러닝 앙상블에서 grid search를 통해 적절한 `max_df`와 `min_df` parameter들을 찾아보았고 그 결과 `max_df`는 0.57, `min_df`는 0.09로 설정하였습니다.

한편 언어모델을 사용하는 경우에는 Wikipedia나 뉴스와 같이 다양한 데이터를 이용해 pretraining된 transformer들을 다운로드해서 활용하였습니다. Hugging Face<sup>3</sup>에서 공개한 [transformer](https://huggingface.co/transformers/master/index.html) 중 일부를 fine-tuning 해보았는데요,

* DistilBERT
* BERT
* OpenAI GPT2 (distilgpt-2)
* XLNet

위 4가지 transformer들을 학습시켜본 결과 입력할 수 있는 문자열의 길이에 제한이 없는 XLNet이 단연 압도적인 성능을 나타냈으나 이에 준하는 압도적인 연산자원을 필요로 했기에 이어서 서술할 cross-validation을 통해 정확한 결과치를 얻지는 못했습니다. 저희는 눈물을 머금고 그 다음으로 성능이 좋았던 GPT2 언어모델을 사용해 실험을 진행하였습니다. GPT2의 경우에도 원본 모델은 굉장히 많은 연산자원을 소모하기에 원본 대신 경량화한(distilled) 버전을 사용하였습니다.

입력 문자열의 최대 길이는 1024로 설정했고, 언어모델의 penultimate layer의 출력값을 얻은 뒤 문자열 방향으로 평균을 취해서 포스트의 임베딩으로 활용하였습니다. 즉 언어모델의 penultimate layer에서 (batch_size x 1024 x model_size) 크기의 3차원 벡터를 얻은 뒤 2번째 차원에 대해 평균을 취해 (batch_size x model_size)의 임베딩을 구했습니다. 이 임베딩은 1개 레이어의 선형변환(linear transformation), 또는 행렬연산을 거쳐 softmax 함수로 전달해주었습니다. 행렬연산에 앞서 dropout이나 layer normalization 등의 추가적인 테크닉도 실험해보았으나 결과에 큰 영향이 없어 전부 제외하였습니다. 보다 더 자세한 정보를 원하신다면 [DIYA 동아리 저장소](https://gitlab.diyaml.com/nlp2020/mbti-kaggle)에서 직접 확인해보실 수 있습니다.

[실험 결과](#experiments)에서는 One-hot encoding을 이용한 BoW 방법을 사용한 경우 CountVectorizer, transformer 언어 모델을 사용한 경우 LanguageModel로 표기하였습니다.

Vectorization | 임베딩 방법
--- | ---
CountVectorizer | scikit-learn의 `CountVectorizer` 이용
LanguageModel | Hugging Face의 `distilgpt-2` 체크포인트 이용


### 2.3. Classification Algorithm

마지막으로 저희는 고전적인 머신러닝 알고리즘과 딥러닝 알고리즘의 성능 차이를 확인해보고자 여러가지 분류기(classifier)를 구현해보았습니다. 고전적인 머신러닝이란 GPU가 상용화되기 이전에 주로 활용되던 CPU 위주의 머신러닝 알고리즘들을 말합니다. 저희가 사용한 머신러닝 알고리즘들은 다음과 같습니다.

Algorithm | Stacking Weight
-- | --
K-Nearest Neighbor | 0.1289
Linear Stochastic Gradient Descent | 0.3840
Logistic Regression | 0.5800
Naive Bayes | 0.4042
Passive Aggressive Classifier | 0.5255
Random Forest | 0.5636
Support Vector Machine (linear kernel) | 0.5641
Gradient Boosting | 0.6314

여기서 stacking weight이란 여러가지 머신러닝 알고리즘을 중첩하는 앙상블(ensemble) 기법의 하나인 stacking에서 각 알고리즘이 차지한 중요도를 의미합니다. 각 알고리즘의 출력값 간 상관계수(correlation)이 크지 않다면 한 알고리즘이 상대적으로 취약한 부분을 다른 알고리즘에서의 출력값을 참고해 보완할 수 있는데요, 이렇게 여러 알고리즘을 함께 활용하는 방법을 앙상블이라고 부릅니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/ensemble-methods-boosting-bagging-stacking.png" style="width: 100%; height: 100%;"/>
<figcaption><font size=2pt>그림 11. 3가지 앙상블 기법<sup>4</sup></font></figcaption>
</center>

앙상블 기법의 종류는 위 이미지에서 보실 수 있듯 크게 Boosting, Bagging, 그리고 Stacking의 3가지 기법으로 나눌 수 있습니다. Boosting이란 한 알고리즘이 큰 오차를 가지는 데이터들을 모아 다른 알고리즘에게 전달해주는 방법을 말합니다. Bagging이란 모든 알고리즘의 출력값을 모아 평균, 최빈값과 같은 대표값을 취하는 방법입니다. 그리고 저희가 이번 실험에서 활용한 Stacking이란 모든 알고리즘의 출력값을 모아 또 다른 알고리즘의 입력값으로 전달해주는 방법인데요, 데이터셋의 일부가 아닌 알고리즘의 출력값들을 전달해준다는 점에서 Boosting과 차이가 있습니다.

이 3가지 기법 중 일반적으로 Stacking이 가장 많은 연산자원을 필요로 합니다. 고전적인 머신러닝 알고리즘들은 연구자들이 사용 가능한 연산자원이 지금보다 훨씬 적을 때 개발된 알고리즘들이기에, 일반적으로 연산자원을 적게 소모하는 대신 딥러닝 알고리즘에 비해 좋지 않은 성능을 보입니다. 따라서 저희는 머신러닝으로도 최대한 딥러닝과 가까운 양의 자원을 소모해야 성능의 동등한 비교가 되리라 생각했고, 이에 다양한 장단점을 가진 알고리즘들을 모아 Stacking 기법을 사용해 실험을 진행해보았습니다.

Stacking을 위한 추가적인 classifier로는 이미 입력값으로 사용하고 있는 Logistic Regression을 한번 더 활용하였습니다. Stacking을 하기 전 먼저 데이터를 분류하는 8개의 분류기(weak learner)들은 모두 각 MBTI 유형에 속할 확률값(0~1)을 출력하도록 했고, 이 값들을 다시 Logistic Regression에 넣어주었습니다. 위 테이블에서의 stacking weight이란 이 logistic regression에서의 weight matrix의 각 열을 분리한 뒤 Euclidean norm을 각각 계산한 결과입니다 (multi-class classification 기준).

Random Forest, Gradient Boosting과 같이 이미 앙상블을 활용하고 있는 강력한 알고리즘들이 Stacking에서도 큰 비중을 차지하고 있음을 확인할 수 있습니다. 한편 Linear Stochastic Gradient Descent는 작은 비중을 차지한 반면 Logistic Regression과 Support Vector Machine(SVM)은 상대적으로 큰 비중을 차지했다는 점을 보면 선형 경계선에서 벗어난 극단치(outlier)들이 분류에 큰 영향을 주고 있다고 유추해볼 수 있겠습니다.

Random Forest, SVM과 같은 각 알고리즘의 의미와 구현 방법에 대해 알아보는 것도 정말 유익한 주제가 되겠지만 이번 주제인 자연어 처리와는 직접적인 관련이 없으므로 이번 포스트에서는 설명을 생략하도록 하겠습니다. 이번 실험에서 가장 weight가 컸던 Gradient Boosting에 대한 설명은 [DIYA의 다른 포스트](https://blog.diyaml.com/teampost/XGBoost-%EB%BF%8C%EC%88%98%EA%B8%B0!/)에 아주 잘 정리되어 있으니 관심 있으신 분들께선 참조해주세요.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/mlp3.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 12. 실험에 사용한 3-layer MLP</font></figcaption>
</center>

저희는 딥러닝 알고리즘으로 위 이미지와 같은 다층 퍼셉트론(Multi-Layer Perceptron, MLP)을 경사하강법(gradient descent)로 학습시키는 방법을 선택했습니다. MLP 모형의 기본구조는 Linear - BatchNorm - Dropout - ReLU를 하나의 레이어로 정하고, 레이어 3개를 이어붙인 다층구조입니다. 이때 행렬연산(Linear)의 입력 차원과 출력 차원은 multi-class classification인지, binary classification인지에 따라 변경하였습니다. Dropout의 경우 CountVectorizer를 이용한 multi-class classification에서 grid search를 통해 dropout probability=0.1로 결정했습니다. 자세한 구현내용은 [DIYA 동아리 저장소](https://gitlab.diyaml.com/nlp2020/mbti-kaggle)를 참조해주세요.

MLP를 3층 이상으로 더 쌓아보기도 했지만 성능에 유의미한 차이는 없었습니다. 따라서 [실험 결과](#experiments)에서 CountVectorizer를 사용한 경우 위 이미지와 같은 3층 MLP를 사용했고, 언어모델을 사용하는 경우 오히려 분류기는 단 하나의 Linear만 남기는 편이 더 높은 성능을 나타내었기에 LanguageModel에서는 Linear 하나만 사용하였습니다. 참고로 LanguageModel에서는 임베딩의 크기가 커서 GPU를 쓰지 않는 Classical ML 알고리즘들로는 학습이 어려워 이 두 개의 조합(LanguageModel + Classical ML)은 실험에서 제외했습니다.

Vectorization | Classifier | 실제 분류 알고리즘
--- | --- | ---
CountVectorizer | Classical ML | 8개 weak learner stacking
CountVectorizer | MLP | 3층 MLP (batch normalization, dropout 사용)
LanguageModel | MLP | Linear 연산만 사용


## 3. <a name="experiments">Experimental Results</a>

앞서 소개한 전처리 방법(Original, Masked, Hypertext), 임베딩 방법(CountVectorizer, LanguageModel), 그리고 분류 알고리즘(Classical ML, MLP)에 따른 학습 결과입니다. Multiclass Classification이란 포스트를 통해 전체 16개 MBTI 유형 중 어디에 속하는지를 맞히는 분류 문제, Binary Classification이란 MBTI의 각 카테고리 중 하나만 맞히는 이진 분류 문제를 뜻합니다. 즉 Multiclass의 경우 분류 알고리즘의 출력값이 intj, esfp 등 전체 MBTI를 맞혀야 정답이고, Binary의 경우 주어진 카테고리에 따라 E인지 I인지, 또는 S인지 N인지만 맞히면 됩니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/mbti/300px-LOOCV.gif" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 13. Leave-One-Out Cross-Validation (n=8)</font></figcaption>
</center>

아래의 결과들은 MBTI 데이터셋을 임의로 3 부분으로 쪼갠 뒤, 이 중 2개 부분으로 학습한 뒤 나머지 1개 부분으로 검증(validate)하는 과정을 각 부분에 대해 한번씩 반복하는 Leave-One-Out Cross-Validation을 수행한 결과입니다. 즉 간단히 말하면 데이터의 2/3로 학습, 1/3로 검증을 수행하고 이를 3번 반복해 평균치를 구한 것입니다. 3개보다 더 잘게 쪼개면 더 많은 양의 데이터로 학습할 수 있다는 장점이 있지만, MBTI 유형 하나당 100개가 안 되는 데이터로는 검증을 하기에 불충분하다고 생각해 3개로 교차검증(cross-validation)을 수행했습니다.


### 3.1. Multiclass Classification

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.6778        | 0.6665        |
| Original  | CountVectorizer    | MLP          | 0.6016        | 0.5747        |
| Original  | LanguageModel      | MLP          | **0.7037**    | **0.6939**    |
| Masked    | CountVectorizer    | Classical ML | 0.4854        | 0.4476        |
| Masked    | CountVectorizer    | MLP          | 0.4360        | 0.4058        |
| Masked    | LanguageModel      | MLP          | **0.4958**    | **0.4718**    |
| Hypertext | CountVectorizer    | Classical ML | 0.4889        | 0.4508        |
| Hypertext | CountVectorizer    | MLP          | 0.4432        | 0.4107        |
| Hypertext | LanguageModel      | MLP          | **0.4946**    | **0.4709**    |

우선 전처리 과정에서 본문에 노출되어있던 MBTI 유형을 가리고 나니(Original -> Masked) 모든 알고리즘에서 거의 20%p 정도 정확도가 하락했음을 알 수 있습니다. enxp와 같이 한두개의 카테고리만 가린 형태는 제거하지 않았다는 점을 고려해볼 때, 만일 글쓴이가 본인의 MBTI 유형을 아예 언급하지 않으려 한다면 이번 실험과 같은 방법으로는 50%대의 정확도를 달성하기는 어려울 것으로 생각됩니다.

한편 유튜브 영상 제목과 같은 추가적인 정보를 넣어주는 경우(Masked -> Hypertext) BoW 기반 알고리즘들의 성능은 오르고, 언어모델을 사용한 딥러닝 알고리즘의 경우 큰 변동이 없는 점을 확인할 수 있습니다. 언어모델의 경우 pretraining 과정에서 저희가 추가한 \<hypertext>라는 토큰을 사전에 학습하지 않았고, 이에 fine-tuning 때에도 다른 토큰들과의 관계성을 제대로 반영하지 못했기 때문으로 보입니다. 그러나 이후 Binary Classification에서는 때로 성능을 높이는 점으로 볼 때 언어모델의 성능에 악영향을 미치지는 않는 것 같습니다.

임베딩의 경우 모든 조합에서 언어 모델을 사용한 경우가 그 성능이 눈에 띄게 좋았습니다. 특히 같은 CountVectorizer를 사용한 임베딩에서 MLP가 Classical ML보다 낮은 성능을 보이고 있기에 언어 모델의 높은 성능이 단순히 딥러닝 알고리즘을 사용한 데서 비롯하지는 않은 것으로 보입니다. 즉 유의미한 임베딩이 알고리즘의 성능을 높인 것이죠. 그러나 언어모델은 pretraining을 통해 Wikipedia와 같은 추가 데이터까지 활용했다는 점을 고려한다면 언어모델이 좋은 임베딩을 제공했을 뿐 아니라 간접적으로 더 많은 말뭉치로부터 학습할 수 있도록 도와주었기 때문이라 볼 수도 있겠습니다.


### 3.2. Binary Classification

Binary Classification의 성능도 대체로 Multiclass와 동일한 추세를 나타냈습니다. 그러나 T와 F를 구별하는 문제에서는 특이하게도 언어모델을 사용하지 않는 편이 오히려 성능이 더 좋았습니다. 이는 T인 사람들과 F인 사람들이 사용하는 어휘의 빈도에서는 차이가 나지만 구사하는 문장의 형태에는 큰 차이가 없음을 의미합니다.

#### 3.2.1. Extraversion (E) vs. Introversion (I)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.8587        | 0.8489        |
| Original  | CountVectorizer    | MLP          | 0.8303        | 0.8119        |
| Original  | LanguageModel      | MLP          | **0.8876**    | **0.8800**    |
| Masked    | CountVectorizer    | Classical ML | 0.8038        | 0.7747        |
| Masked    | CountVectorizer    | MLP          | 0.7909        | 0.7660        |
| Masked    | LanguageModel      | MLP          | **0.8146**    | **0.7904**    |
| Hypertext | CountVectorizer    | Classical ML | 0.8034        | 0.7745        |
| Hypertext | CountVectorizer    | MLP          | 0.7913        | 0.7664        |
| Hypertext | LanguageModel      | MLP          | **0.8161**    | **0.7937**    |

#### 3.2.2. Sensing (S) vs. Intuition (N)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.9132        | 0.9050        |
| Original  | CountVectorizer    | MLP          | 0.8924        | 0.8720        |
| Original  | LanguageModel      | MLP          | **0.9251**    | **0.9148**    |
| Masked    | CountVectorizer    | Classical ML | 0.8660        | 0.8193        |
| Masked    | CountVectorizer    | MLP          | 0.8625        | 0.8313        |
| Masked    | LanguageModel      | MLP          | **0.8719**    | **0.8557**    |
| Hypertext | CountVectorizer    | Classical ML | 0.8644        | 0.8178        |
| Hypertext | CountVectorizer    | MLP          | 0.8610        | 0.8290        |
| Hypertext | LanguageModel      | MLP          | **0.8739**    | **0.8581**    |

#### 3.2.3. Thinking (T) vs. Feeling (F)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.8583        | **0.8583**    |
| Original  | CountVectorizer    | MLP          | 0.8383        | 0.8389        |
| Original  | LanguageModel      | MLP          | **0.8589**    | 0.8557        |
| Masked    | CountVectorizer    | Classical ML | **0.7899**    | **0.7898**    |
| Masked    | CountVectorizer    | MLP          | 0.7818        | 0.7827        |
| Masked    | LanguageModel      | MLP          | 0.7703        | 0.7622        |
| Hypertext | CountVectorizer    | Classical ML | **0.7957**    | **0.7956**    |
| Hypertext | CountVectorizer    | MLP          | 0.7746        | 0.7756        |
| Hypertext | LanguageModel      | MLP          | 0.7756        | 0.7706        |

#### 3.2.4. Judging (J) vs. Perceiving (P)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.8026        | 0.8008        |
| Original  | CountVectorizer    | MLP          | 0.7530        | 0.7501        |
| Original  | LanguageModel      | MLP          | **0.8524**    | **0.8494**    |
| Masked    | CountVectorizer    | Classical ML | 0.7197        | 0.7104        |
| Masked    | CountVectorizer    | MLP          | 0.6898        | 0.6848        |
| Masked    | LanguageModel      | MLP          | **0.7274**    | **0.7179**    |
| Hypertext | CountVectorizer    | Classical ML | 0.7135        | 0.7047        |
| Hypertext | CountVectorizer    | MLP          | 0.6830        | 0.6772        |
| Hypertext | LanguageModel      | MLP          | **0.7320**    | **0.7224**    |


### 3.3. 한계점 및 결론

이상의 실험 결과를 통해 저희는 다양한 전처리 방법과 임베딩 방법이 성능에 유의미한 영향을 줄 수 있다는 사실을 확인할 수 있었습니다. 전처리의 경우 본문에 드러나있는 MBTI 레이블들이 현재 [Kaggle에 있는 코드들](https://www.kaggle.com/datasnaek/mbti-type/code)의 성능에 큰 영향(~20%p)을 주고 있다는 점을 알 수 있었고, hypertext를 대체하면 BoW 알고리즘의 성능을 향상시킬 수 있다는 점을 관찰할 수 있었습니다. 그러나 hypertext가 언어모델을 이용한 알고리즘에 미치는 영향을 파악하기 위해서는 언어모델의 fine-tuning을 조금 더 오래 해볼 필요가 있어보입니다.

CountVectorizer와 언어모델을 비교해보면 언어모델이 가져다주는 성능 향상이 매우 큰 것으로 나타났습니다. 그러나 앞서 언급한 바와 같이, 이 성능 향상이 데이터의 추가가 아닌 문맥적 의미의 이해로부터 오는지를 알기 위해서는 pretraining이 되지 않은 상태의 언어모델을 이용해 같은 실험을 반복해보아야 할 것입니다. 이때 pretrained token embedding도 사용하지 않는다면 CountVectorizer에서 나타난 Classical ML과 MLP 간의 성능 차이도 딥러닝 모델의 복잡도(complexity)를 높이면 극복할 수 있는지 확인해볼 수 있으리라 생각합니다.

Classical ML이 CountVectorizer에서는 MLP보다 뛰어난 성능을 나타냈는데, 언어모델을 통해 임베딩을 얻는 LanguageModel에서는 그 성능을 측정하지 못했습니다. 최근 다양한 머신러닝 알고리즘에서 GPU를 이용해 빠른 시간 내에 근사해(approximate solution)을 구하는 방법들이 개발되고 있는데, 이러한 방법들을 이용하면 LanguageModel과도 결합해 더 좋은 성능을 낼 수 있으리라 기대합니다. 그리고 언어모델 자체 역시 마찬가지로, 만약 연산자원이 더 주어졌다면 더 큰 메모리 용량과 계산 복잡도를 가진 언어모델을 활용해 더욱 성능을 높여볼 수 있었으리라 생각합니다.

마지막으로, 본래 목표했던 고객 집단 분류의 측면으로 보면 이번 실험과 같이 분석을 진행한다면 순수 자연어 데이터만으로는 고객층의 분류가 쉽지 않을 것으로 생각됩니다. E vs. I 또는 S vs. N 처럼 고객층의 이분법적인 특징을 먼저 정의한 뒤 그 특징만을 분류하거나, 고객 간 상호작용이나 실질적인 서비스 이용 내역 등을 필수적으로 활용해야할 것으로 보입니다.


## References

##### <sup>1</sup><sub>[Gupta, V. K. (2014). Role of Crm in Marketing](https://www.researchgate.net/publication/240296261_Expanding_the_Role_of_Marketing_From_Customer_Equity_to_Market_Capitalization)</sub>
##### <sup>2</sup><sub>[scikit-learn: machine learning in Python — scikit-learn 0.24.2 documentation](https://scikit-learn.org/stable/)</sub>
##### <sup>3</sup><sub>[Hugging Face – The AI community building the future.](https://huggingface.co/)</sub>
##### <sup>4</sup><sub>[Ensemble Methods: dé 3 methoden eenvoudig uitgelegd](https://pythoncursus.nl/ensemble-methods/)</sub>


---
Written by 김영훈, 이준후, 정은기, 정은빈, 홍원기