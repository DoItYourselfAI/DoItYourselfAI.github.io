---
layout: article
title: "Searching for Neural Architectures"
categories: teampost
last_modified_at: 2020-01-17T13:00:00+09:00
tags: [automl]
external_teaser: "https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/darts_cell_design.jpg"
external_feature: "https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/darts_cell_design.jpg"
excerpt: "NAS의 기능을 살펴보고, Evolving Neural Networks through Augmenting Topologies (NEAT) 와 Differentiable Architecture Search (DARTS) 를 직접 구현해보았습니다."
---


## 목차
1. [Introduction](#intro)
2. [Evolving Neural Networks through Augmenting Topologies (NEAT)](#neat)
3. [Differentiable Architecture Search (DARTS)](#darts)
4. [Experiments](#experiments)
5. [Conclusion](#concl)


## <a name="intro">1. Introduction</a>

Neural Architecture Search (NAS)란 특정 문제를 해결하기 위한 가장 적합한 신경망 구조의 종류 및 형태를 딥러닝을 통해 탐색하는 방법을 말합니다. 본 포스트에서는 NAS의 기능을 살펴보고 Evolving Neural Networks through Augmenting Topologies(NEAT)와 Differentiable Architecture Search(DARTS)를 직접 구현해 본 내용을 담고 있습니다.

>학습과 구현은 [KC-ML2](https://www.kc-ml2.com)의 자원으로 ML2의 컴퓨팅 서버를 활용하였습니다.

우선 NAS의 기능을 아래와 같은 2가지 관점으로 나누어 살펴보고자 합니다.<sup>[1](#survey)</sup>

(1) Search Space Perspective <br>
(2) Optimization Perspective


### (1) Search Space Perspective

먼저 인공신경망의 구조를 탐색하기 위해서는 인공신경망이 어떻게 구성되어있는지, 즉 신경망의 최소 구성 단위에 대해 정의해주어야 합니다. NAS에서의 신경망은 search space라고 부르는, 사전에 정의한 연산자 및 함수들로 구성된 primitive operations들을 선택 및 조합하여 생성됩니다. 이때 구체적인 연산자의 예시로는 크게 convolution, pooling, concatenation, element-wise addition, skip connection 등등이 있습니다.

Primitive operations들을 정의한 후, 이로부터 신경망을 만드는 데에는 여러 가지 방법이 있는데요, 여기에서는 search space의 크기에 따라 분류한 4가지 방법들을 간단히 소개해보고자 합니다.

- Entire Structure Search
- Cell-based Structure Search
- Hierarchical Structure Search
- Network Morphism based Structure Search


#### Entire Structure Search

Entire structure search란 가장 직관적인 방법으로서, primitive operations들의 임의적인 조합을 통해 신경망의 전체 구조를 한번에 생성해보는 방법을 말합니다. 하지만 layer가 많아질수록 search space가 거대해져 가장 좋은 신경망 구조를 찾는데 많은 시간과 컴퓨팅 자원이 필요하고, 또한 NAS로 생성하게 되는 network의 기본 단위가 매우 커지게 되어 서로 다른 데이터에 같은 신경망 구조를 적용하고자 할 때 transferability가 떨어질 수 있다는 단점이 있습니다.


#### Cell-based Structure Search

Cell-based structure search란 먼저 신경망의 일부분에 해당하는 cell structure를 찾은 후, 이를 사전에 정의한 숫자만큼 쌓아 전체의 neural network를 구성하는 방식을 뜻합니다. 이때 cell-based structure search는 다시 cell 내에서의 operation 종류 및 connection의 선택을 뜻하는 *cell-level search* 와 이렇게 결정된 cell들의 개수 및 배치를 선택하는 *network-level search* 로 나누어볼 수 있습니다.

위 방법은 entire structure search에 비해 search space의 크기를 현저히 줄일 수 있다는 장점이 있습니다. 또한 기본 단위가 cell이 되며, 이 cell을 더 쌓는 방식을 통해 작은 데이터셋 보다 큰 데이터셋으로 이전(transfer)하기가 쉽기 때문에 transferability가 훨씬 좋습니다. 하지만 cell structure가 한번 결정되고 나면, *network-level search* 단계에서는 cell을 단순히 쌓거나 재배열하는 방식으로 전체의 신경망을 생성하기에, network level에서는 복잡한 구조를 생성하기 어렵다는 단점을 가지고 있습니다.


#### Hierarchical Structure Search

앞서 말씀드린 cell-based structure search 방법의 단점을 보완하기 위해, hierarchical structure search 방법에서는 다양한 level에서 cell를 생성하고자 합니다. 이때 higher-level cell은 lower-level cell들을 반복적으로 통합하여 만들어지는데요, 가장 상위 level의 cell은 전체 신경망 구조와 같습니다.

Hierarchical structure search는 cell-based structure search보다  search space가 복잡한 대신 보다 유연하게 구조를 생성할 수 있어 다양한 구조와 형태를 발견할 수 있습니다.


#### Network Morphism based Structure Search

Network morphism based structure search란 현재 널리 사용되고 있는 잘 알려진 신경망 (reference model)에 담긴 형태 및 연결 정보를 추출하여 새로운 신경망을 만드는 방법입니다.

앞서 언급했던 3가지 방식들은 신경망의 구조를 백지에서부터 생성해나가고자 했던 반면 이 방법에서는 이미 성능이 보장되어 있는 reference model에 기초하여 구조를 형성해나가기에, reference model에 비해 더 좋거나 최소한 동등한 수준의 성능을 보장한다는 장점이 있습니다.


### (2) Optimization Perspective

신경망을 이루는 search space를 정의한 이후에는 이로부터 만들어낼 수 있는 조합 중 가장 높은 성능을 가진 구조를 찾아야할 것입니다. 이 과정은 결국 연산자의 선택, 그리고 선택된 연산자 간의 connection을 최적화하는 일이라고 볼 수 있는데요, 이는 딥러닝 분야에서 hyperparameter optimization (HPO)이라고 부르는 과정과 유사합니다. 실제로 NAS에서는 최적의 신경망 구조를 찾기 위해 HPO에서 활용하는, 아래와 같은 여러가지 최적화 기법을 활용합니다.

- Grid & Random Search
- Evolution-based Algorithm
- Reinforcement Learning
- Differentiable Architecture Search


#### Grid & Random Search

Grid search와 random search는 HPO에서 가장 흔하게 사용하는 방법입니다. Grid search는 search space를 일정한 간격(grid)으로 나눠 신경망 구조를 grid에 해당되는 모든 값들로 각각 한번씩 훈련시켜봄으로써, 가장 좋은 조합을 찾고자 하는 방법입니다.

Grid search는 병렬로 탐색을 진행하기 쉽고, 같은 시간 동안 사람이 일일이 조합을 하나씩 해보는 것보다 그 성능이 우수한 경향이 있어서 자주 사용됩니다. 하지만 grid에 해당하는 점들만 테스트해볼 수 있다는 점, 그리고 중요하지 않은 일부 영역 또한 중요한 영역과 동등하게 탐색하기에 불필요하게 많은 시도를 해봐야한다는 점 등이 단점으로 꼽힙니다.

Random search는 정말 각 조합을 임의적으로 선택해보는 것입니다. Random search는 그 횟수만 많다면 grid search보다 더 다양한 조합을 탐색할 수 있으며 이론적으로, 그리고 실험적으로 grid search보다 실용적이고 효율적임이 밝혀졌습니다. 하지만 탐색 끝에 최종적으로 선택된 조합이 정말 최선인지에 대하여 확인하는 일이 매우 어려우며, 최적 조합을 찾는데 많은 시간과 컴퓨팅 자원이 필요하다는 단점이 있습니다.

이때 random search의 성능과 필요로 하는 자원 간의 trade-off를 조절하기 위해 만들어진 알고리즘이 바로 Hyperband입니다. Hyperband에서는 학습 도중 model의 성능이 하위권에 해당하는 경우 이를 버리고, 여기에 투입되었던 자원을 보다 가능성이 있다고 생각되는 조합에 추가로 할당합니다.


#### Evolution-based Algorithm

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/EB_Overview.png" alt="진화 알고리즘의 4가지 단계" style="width:70%; height:70%"/>
    <figcaption><font size=2pt>
    	진화 알고리즘의 4가지 단계
    	<br><i> Source: <a href="https://arxiv.org/pdf/1908.00709.pdf">AutoML: A Survey of the State-of-the-Art</a>, He et al, arXiv preprint 2019.</i>
    </font></figcaption>
  </center>
</figure>

Evolution-based algorithm (진화 알고리즘, 이하 EA라고 명칭)은 생물학적 진화로부터 영감을 얻은 알고리즘입니다. 수식적으로 최적해를 유도하는 calculus-based method, 가능한 모든 해를 탐색해보고자 하는 exhaustive method들과 달리 EA는 한 번에 여러개의 해를 탐색하되, 각각의 해 근처에서 최적화를 진행하는 population-based method의 일종입니다.

EA는 다음 4 가지 단계로 구성됩니다.

1. Selection Step
2. Crossover Step
3. Mutation Step
4. Update Step


##### Selection Step

Selection step은 이후의 3 가지 step을 진행하기 위한 후보군을 선택하는 과정입니다. NAS의 경우 이 후보란 특정한 신경망 구조 (연산자들과 그 연산자들 간의 연결 구조)가 되는데요, 각각의 구조가 우리가 풀고자 하는 문제에 얼마나 적합한지를 수치로 표현한 적합도 (fitness)를 이용해 다음과 같은 방법으로 후보군을 선택할 수 있습니다.

- **Fitness Selection**: 각 구조를 선택할 확률이 적합도의 절대적인 크기에 비례하도록 선택하는 방법입니다.
- **Rank Selection**: 각 구조를 선택할 확률이 적합도의 순위에 비례하도록 선택하는 방법입니다.
- **Tournament Selection**: 가장 널리 사용되는 선택 방법으로서 먼저 각 구조의 적합도에 따라 정렬한 뒤, 적합도가 가장 높은 구조를 선택할 확률이 $p$, 2번째로 높은 구조를 선택할 확률이 $p(1-p)$, $n$번째로 높은 구조를 선택할 확률이 $p(1-p)^{n-1}$가 되도록 선택하는 방법입니다.

##### Crossover Step

앞선 selection step에서 선택한 후보군들 중 임의로 2개의 후보를 뽑아 둘의 특성을 모두 공유하는 자손 (offspring)을 만드는 단계입니다. 이는 생물학적 교배, 교차와 유사합니다.

##### Mutation Step

Crossover step에서 생성한 자손이 가지는 연산자 및 구조에 조금의 돌변변이를 만드는 과정입니다. 이를 통해 기존의 후보군에는 존재하지 않았던 새로운 형질을 만들어냄으로써 다양한 해를 탐색해볼 수 있습니다.

##### Update Step

Update step에서는 crossover step과 mutation step에서 생성한 구조들을 다음 selection step에서 후보군으로 선택할 수 있도록 새로 메모리나 연산자원을 할당해줍니다. 이후 다시 selection step으로 돌아가 앞선 4가지 단계를 반복합니다.


### Reinforcement Learning

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/RL_Overview.png" alt="강화학습에서 agent와 환경간의 상호작용" style="width:70%; height:70%"/>
    <figcaption><font size=2pt>
    	강화학습에서의 agent와 환경 간 상호작용
    	<br><i> Source: <a href="https://arxiv.org/pdf/1908.00709.pdf">Same as above.</a></i>
    </font></figcaption>
  </center>
</figure>

NAS에서의 최적화는 강화학습(Reinforcement Learning)을 통해 이루어지기도 합니다. 강화학습에서는 환경과의 상호작용을 통한 agent의 학습을 목표로 하는데요, NAS에서의 환경이란 특정 신경망 구조가 얼마나 적합한지를 판단해주는 평가기준으로 생각할 수 있습니다. NAS에서의 agent는 상호작용이 이루어지는 매 시기 $t$마다 주어진 search space 상에서 특정한 신경망 구조를 선택하는 행동 $A_t$을 수행하여, 환경으로부터 해당 구조의 적합도를 보상 $R_t$로 받습니다. Agent는 이 보상 신호를 통해 가장 적합한 행동의 종류 및 그 순서를 학습하게 됩니다.

이때 강화학습을 기초로한 NAS 알고리즘은 CIFAR10과 PTB 데이터셋에서 state-of-the-art에 해당하는 성능을 기록했으나, 학습에 필요한 시간과 자원이 많이 필요하여 개인의 연구 및 활용에는 부적합하다는 평이 많았습니다. 그러나 최근 이러한 문제를 해결하기 위해 분산 컴퓨팅을 활용하거나 강화학습의 각 시도마다 학습시킨 신경망의 모수들을 저장해둔 뒤 재활용하는 등의 여러가지 시도들이 이루어지고 있습니다.<sup>[1](#survey)</sup>


### Differentiable Architecture Search

앞서 소개드렸던 3가지 방법들은 모두 search space에서 한 가지 구조를 샘플링할 때 이를 이산적인 (discrete) 표본으로 간주합니다. 즉 search space 상에서 한 표본을 뽑아 이를 학습시켜본 뒤 결과를 관측할 때, 해당 표본의 결과는 근처에 존재하는 다른 표본들의 예측치에 대해 영향을 주지 않습니다.

이와 달리 Differentiable Architecture Search (DARTS)는 search space를 연속적이고 미분가능한 (differentiable) 공간으로 가정한 알고리즘입니다. DARTS에서는 가능한 모든 연산자와 연결구조를 softmax 함수를 이용, 하나의 커다란 신경망 안에 전부 집어넣습니다. 이후 학습 과정에서 미분계수 (gradient)의 크기에 따라 각각의 연산자가 차지하는 비중을 조절해나가면서, 학습을 마칠 때 가장 많은 비중을 차지하는 연산자만을 남기는 방식으로 신경망을 구성합니다.

DARTS는 한번에 전체 search space를 탐색하고자 하므로 탐색 시간을 크게 감소시킬 수 있지만, search space의 크기에 따라 필요한 메모리가 선형적으로 (또는 제곱으로) 증가하고, 신경망 구조가 깊어지면 학습 과정에서 선택되는 연산자의 종류가 단순해진다는 단점이 있습니다.

NAS 알고리즘은 앞서 말씀드렸던 4가지 search space perspective와 4가지 optimization perspective의 분류에 따라 각각 무수히 많은 알고리즘이 알려져 있는데요, 저희는 이 중 잘 알려져 있으면서도 성격이 크게 다른 2가지 알고리즘을 선택해 직접 구현해보았습니다. 하나는 entire structure search를 목표로 하는 evolution-based algorithm인 NEAT, 그리고 다른 하나는 cell-based structure search를 목표로 하는 differentiable architecture search 방식인 DARTS입니다. 이후 섹션에서는 저희가 해당 알고리즘에 맞추어 search space를 정의한 방법과 더불어 알고리즘에 대한 더 자세한 설명을 드리도록 하겠습니다.


## <a name="neat">2. Evolving Neural Networks through Augmenting Topologies (NEAT)</a>

NeuroEvolution of Augmenting Topologies (NEAT)<sup>[2](#evolving)</sup>는 gradient를 이용하지 않는, evolution-based algorithm의 일종입니다. NEAT 알고리즘에서는 node와 connection으로 이루어지는 그래프 구조를 search space로 정의하고, 이 그래프를 확장해나가는 방법으로 entire structure search를 수행하는 것을 목표로 합니다.

### Genome

생물학적 진화로부터 영감을 얻은 알고리즘인 만큼, 신경망에서의 node와 connection은 genome이라고 불리는 정보로부터 형성됩니다. Genome은 node gene과 connection gene으로 나누어지는데요, node gene은 주로 연산자의 종류를 결정해주고 connection gene은 각 연산자끼리의 연결 구조를 정의해줍니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/Genome.png" alt="Genome 예시"/>
    <figcaption><font size=2pt>
    	Node gene 및 connection gene의 예시와 이를 이용한 신경망 구조
    	<br><i> Source: <a href="http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">Evolving Neural Networks through Augmenting Topologies</a>, Stanley & Miikkulainen, Evolutionary computation 10(2), 2002.
    </i></font></figcaption>
  </center>
</figure>

EA에서의 selection step은 현재 단계에서 가지고 있는 이러한 genome들 중 일부를 선택하는 방식으로 이루어지게 됩니다.

### Mutation

EA 알고리즘에서 다양한 네트워크 구조를 만들기 위해서는 crossover step과 mutation step을 수행해야 합니다. NEAT에서는 mutation을 crossover보다 먼저 수행해주는데요, mutation은 새로운 gene을 생성하거나 기존의 gene을 삭제하는 일을 말합니다. 이때 mutation이 일어날 확률은 알고리즘을 시행하기 전 미리 정의하는 hyperparameter로서, 기본값은 EA의 매 반복당 $p=0.2$로 정의되어 있습니다.

위 그림에서 눈여겨 보아야 할 점은 node gene이 추가되는 경우인데요, 새로운 node가 만약 기존의 connection의 중간에 추가가 되면 기존에 존재하던 node 간의 연결구조가 달라질 수 있습니다. 이 경우 이전에 존재했던 connection은 비활성화되고 (disabled; DIS) 새로운 2개의 connection이 추가됩니다. 예를 들어 node3 $\rightarrow$ node4 사이에 node6가 추가되면 node3 $\rightarrow$ node4는 비활성화되고 node3 $\rightarrow$ node6, node6 $\rightarrow$ node4가 새로 생성됩니다. 이때 비활성화한다는 것은 해당 node가 아예 없었던 것으로 만드는 제거와는 다른 개념인데요, 이 차이는 이어서 언급할 innovation number에서 잘 드러납니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/mutation_method.png" alt="Mutation이 일어나는 과정"/>
    <figcaption><font size=2pt>
    	새로운 node를 추가하거나 connection을 추가하는 mutation 과정
    	<br><i> Source: <a href="http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">Same as above.</a>
    </i></font></figcaption>
  </center>
</figure>

### Innovation Number and Crossover

NEAT에서는 두 genome을 crossover하기 위해 각각의 gene이 처음 mutation을 통해 생겨난 시점, innovation number를 기록합니다. 이 숫자는 NEAT 알고리즘 상에서 생성한 모든 genome이 공유하는 번호 체계로, 서로 다른 genome에서 생겨난 gene들은 모두 고유한 번호를 가지고 있습니다. 예를 들어 EA의 첫번째 세대에서 총 5개의 connection gene이 생겨나 이에 각각 1 ~ 5의 번호를 매긴다고 하면, 두번쨰 세대에서 crossover step과 mutation step을 통해 2개의 새로운 gene들이 생겨났다고 하면 이들의 innovation number는 6, 7이 됩니다. 그런데 만약 중간에 하나의 gene이 비활성화되는 경우, 해당 gene은 genome이 생성하는 구조에서는 사라지게 되지만 그 innovation number는 남아있게 됩니다.

두 genome을 crossover 할 때에는 우선 두 genome의 gene들을 innovation number가 낮은 순에서부터 높은 순으로 정렬합니다. 이후 두 genome의 모든 gene들을 전부 가지는 새로운 genome을 만드는데, 만약 도중에 gene이 비활성화됨 (DIS)으로써 같은 innovation number에 대해 다른 gene을 가지고 있는 경우 두 genome 중 성능이 더 좋은 genome의 활성화 상태를 물려받도록 합니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/crossover.png" alt="두 genome이 만나 새로운 genome을 만드는 crossover 과정"/>
    <figcaption><font size=2pt>
    	두 genome이 만나 새로운 genome을 만드는 crossover 과정
    	<br><i> Source: <a href="http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">Same as above.</a>
    </i></font></figcaption>
  </center>
</figure>


### Cell Design

저희는 NEAT 알고리즘을 이용해 MNIST와 SST-2 데이터셋에서 높은 성능을 낼 수 있는 신경망 구조를 탐색해보았는데요, 이미지 데이터인 MNIST와 자연어처리 데이터인 SST-2를 모두 처리할 수 있도록 NEAT에서의 node gene을 아래와 같이 정의해보았습니다.

우선 이미지 데이터를 처리하는 성능이 뛰어난 것으로 알려진 Residual Network에서 이용하는 skip connection을 구현하기 위해, 하나의 node는 이전의 node 2개 (Node I, Node J)로부터 입력 데이터를 받아 연산의 결과를 이후의 node 1개 (Node K)로 출력하도록 디자인하였습니다. 2개의 입력값은 각각 서로 다른 연산의 대상이 되는데요, 선택 가능한 전체 연산자의 목록은 아래와 같습니다.

* none (no connection)
* identity (skip connection)
* separable convolution (kernel_size: 1, 3, 5)
* dilated convolution (kernel_size: 1, 3, 5)
* average pooling (kernel_size: 3, 5)
* max pooling (kernel_size: 3, 5)
* multi-head attention (num_heads: 8)

이때 MNIST의 경우 attention을 선택 가능한 연산자에서 제외하였고, SST-2에서만 attention을 활용하였습니다.

위의 연산을 취한 뒤에는 아래의 4가지 활성화함수 중 하나를 선택할 수 있도록 node gene를 구성하였는데요, 실제 실험 시에는 search space가 지나치게 커지는 것을 막기 위해 MNIST와 SST-2에서 모두 identity와 ReLU만 활용하였습니다.

* identity (no activation function)
* ReLU
* sigmoid
* hyperbolic tangent

활성화함수까지 지난 2개의 출력값은 다음 노드 (Node K)로 전달하기 위해 elementwise-addition 또는 elementwise-multiplication을 취해줍니다. 그러나 활성화함수의 경우와 마찬가지로 실제 실험 시에는 두 데이터셋에서 모두 언제나 addition만 사용하도록 선택의 폭을 줄여주었습니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/neat_cell_design.png" alt="Cell Design 도식"/>
    <figcaption><font size=2pt>
    	NAS 실험에서 사용한 함수 및 연산의 종류
    	</font></figcaption>
  </center>
</figure>

이렇게 cell을 정의한 뒤, 한 세대 당 50개의 genome을 생성하도록 설정해 데이터셋당 각각 약 12시간 (GPU time) 동안 NEAT 알고리즘을 수행하였는데요, 자세한 hyperparameter는 [DIYA  저장소](https://gitlab.diyaml.com/automl/automl-search)에 올려둔 코드를 참조해주세요 :)


## <a name="darts">3. Differentiable Architecture Search (DARTS)</a>

Differentiable Architecture Search (DARTS)<sup>[3](#darts)</sup>는 경사하강법 (gradient descent)을 기반으로 하는 cell-based structure search 방법입니다. 이 방법의 가장 큰 특징은 softmax 함수를 이용해 search space를 미분가능하도록 만들어주었다는 점입니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/darts_cell_design.jpg" alt="DARTS Cell Design 과정"/>
    <figcaption><font size=2pt>
    	DARTS의 학습 과정에서 각 연산의 가중치가 변화하는 과정
    	<br><i> Source: <a href="https://arxiv.org/abs/1806.09055">DARTS: Differentiable Architecture Search</a>, Liu et al, arXiv preprint 2019.
    </i></font></figcaption>
  </center>
</figure>

DARTS는 하나의 cell의 구조를 결정한 뒤 이 cell을 여러개 쌓아서 전체 신경망 네트워크를 구성하는 NAS 알고리즘으로, 이 cell은 여러 개의 node로 구성된 directed acyclic graph (DAG)로 표현할 수 있습니다. 즉 node들 간에는 순서가 있으며, cell에 입력된 데이터는 0번 node에서부터 가장 큰 번호의 node까지 순서대로 전달됩니다.

저희가 정의한 NEAT에서는 각각의 node 안에서 실질적인 연산이 이루어졌던 반면, DARTS에서는 node가 아닌 각각의 connection이 연산의 종류를 나타냅니다. DARTS 알고리즘을 시작할 때 각 node들은 이전의 모든 node들과 search space 상에서 가능한 모든 연산자 (mixed operation)으로 연결되어있고, 각 node로의 입력값은 이러한 mixed operation들의 weighted sum으로 정의됩니다. 처음에는 각 connection의 가중치를 모두 동일하게 설정하고, 이후 말씀드릴 목적식을 최적화하는 과정을 통해 이 가중치들을 경사하강법으로 점차 변화시켜나갑니다. 학습을 마친 뒤 DARTS 알고리즘은 각각의 node에서 가중치가 가장 큰 입력값 2개씩을 남기고 나머지를 전부 제거합니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/conv_cell_1.png" alt="DARTS Normal Convolutional Cell 예시"/>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/conv_cell_2.png" alt="DARTS Reduction Convolutional Cell 예시"/>
    <figcaption><font size=2pt>
    	DARTS로 학습한 Convolutional Cell 구조
		<br><i> Source: <a href="https://arxiv.org/abs/1806.09055">Same as above.</a>
    </i></font></figcaption>
  </center>
</figure>

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/recurrent_cell.jpg" alt="DARTS Recurrent Cell 예시" style="width:90%; height:90%"/>
    <figcaption><font size=2pt>
    	DARTS로 학습한 Recurrent Cell 구조
    	<br><i> Source: <a href="https://arxiv.org/abs/1806.09055">Same as above.</a>
    </i></font></figcaption>
  </center>
</figure>

### Bilevel Optimization

그런데 DARTS에서와 같이 경사하강법을 통해 신경망 구조를 탐색하기 위해서는 구조를 탐색함과 동시에, 변화하는 구조에 맞춰 신경망의 모수들도 학습시켜주어야 합니다. 따라서 DARTS에서는 학습용 데이터셋을 두 개로 나누어 하나는 모수 $w$의 학습을 위한 training set, 나머지 하나는 연산의 가중치 $\alpha$ 학습을 위한 validation set으로 사용합니다.<sup>[4](#html)</sup>

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/optimization.jpg" alt="DARTS 최적화" style="width:75%; height:75%"/>
    <figcaption><font size=2pt>
    	DARTS의 2중 최적화 목적함수
    	<br><i> Source: <a href="https://arxiv.org/abs/1806.09055">Same as above.</a>
    </i></font></figcaption>
  </center>
</figure>

DARTS 알고리즘은 위에 정의된 2중 최적화 (bilevel optimization) 문제를 풂으로써 모수를 학습시킴과 동시에 각 가중치를 학습시키고자 합니다. 그러나 주어진 $\alpha$에 대해 매번 training loss를 최소화하는 $w^* $를 찾는 일은 많은 시간과 자원을 필요로 하므로, 실제 구현 상으로는 $w^* $ 대신 현재의 $w, \alpha$ 상에서 경사하강을 한번 진행한 값 $w - \xi \nabla_w \mathcal{L}_\text{train}(w, \alpha)$을 이용해 $\alpha$를 학습시킵니다. 이렇게 경사하강의 방향을 근사하게 되면 비록 매 step에서의 학습 방향에는 오차가 존재할 수 있지만 알고리즘의 복잡도를 크게 줄일 수 있습니다. 원 논문에서는 학습 방향의 오차를 줄이기 위해 $w$와 $\alpha$의 2계 미분을 이용하는 2차 근사 방법도 제안하고 있습니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/darts_algorithm.jpg" alt="DARTS 알고리즘"/>
    <figcaption><font size=2pt>
    	경사하강법을 이용한 DARTS 알고리즘
    	<br><i> Source: <a href="https://arxiv.org/abs/1806.09055">Same as above.</a>
    </i></font></figcaption>
  </center>
</figure>

### Cell Design

NEAT 알고리즘과의 비교를 위해 저희는 앞서 정의한 search space를 동일하게 DARTS 알고리즘에도 적용해보았습니다. 그러나 앞서 말씀드렸듯 NEAT에서의 node과 DARTS에서의 node 간 정의에는 차이가 있어, 다음과 같은 방법으로 학습조건을 간접적으로 통일시켰습니다. 우선 각 node는 2개의 node로부터 입력값을 받고 하나의 node로만 출력값을 전달해주도록 하였고, 동일한 연산자와 활성화함수의 집합으로부터 선택하도록 일치시켜준 뒤 DARTS를 수행하였습니다.

비록 저희가 수행한 실험에는 포함되지 않았지만, 만약 두 node로부터 받은 연산값을 elementwise-addition 대신 elementwise-multiplication으로도 합칠 수 있는 경우를 상정하면 각 연산자의 가중치 $\alpha$ 이외의 새로운 가중치를 필요로 하게 됩니다. 이는 각 연산자의 최적화가 이러한 집합 함수 (aggregation function)에 대해 조건부로 이루어져야 하기 때문입니다. 즉 DARTS에서의 목적함수는 최적 모수를 기반으로 한 최적 연산자를 찾은 뒤, 이를 바탕으로 최적 집합 함수를 찾는 3중 최적화 문제가 되는 것이죠. 이렇게 되면 mixed operation에 포함되는 연산의 종류가 집합 함수의 개수만큼 배로 늘게 됩니다.

따라서 저희는 NEAT와 마찬가지로 집합 함수로써 elementwise-addition만을 사용해, node 4개를 가지는 cell 하나를 생성한 뒤 이를 4개 쌓는 방식으로 DARTS 알고리즘을 구현하였습니다. 더 자세한 내용은 [DIYA 동아리 저장소](https://gitlab.diyaml.com/automl/automl-search)를 참조해주세요 :)


## <a name="experiments">4. Experiments</a>

저희는 이미지 데이터셋 MNIST<sup>[5](#MNIST)</sup>와 자연어처리 데이터셋 SST-2<sup>[6](#nlp)</sup>에 대해 NEAT와 DARTS 알고리즘을 구현해보았는데요, 우선 저희가 사전에 정의한 search space가 문제를 해결하는 데에 적합했는지를 확인해보기 위해 search space에서의 연산자와 구조를 이용해 baseline model들을 직접 만들어보았습니다.

### Baselines

Baseline model의 구조를 정의하기 위해서 저희는 각 데이터셋에서의 state-of-the-art (SOTA)에 해당하는 알고리즘을 소개해주는 사이트인 [paperswithcode.com](https://paperswithcode.com/)<sup>[7](#code)</sup>를 참조하였습니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/Paperswithcode.png" alt="paperswithcode.com의 State-of-the-art"/>
    <figcaption><font size=2pt>
    	실험 당시 paperswithcode.com에 공개된 MNIST에서의 SOTA
    	<br><i> Source: <a href="https://paperswithcode.com">paperswithcode.com</a>
    </i></font></figcaption>
  </center>
</figure>

#### MNIST Dataset

저희가 실험을 진행할 당시 MNIST에서의 SOTA는 RMDL<sup>[8](#cifar)</sup> 알고리즘으로 99.82%의 성능을 보이고 있었습니다. RMDL은 여러 형태의 신경망 구조들을 임의적으로 생성해 이들을 전부 학습시킨 뒤 앙상블을 하는 알고리즘인데요, 여러 구조를 생성한다는 점에서 NAS와 매우 유사하나 NEAT 또는 DARTS로는 생성하기 어려운 구조였기에 이 알고리즘 대신 다른 이미지 데이터셋 (CIFAR<sup>[8](#cifar)</sup>, ImageNet<sup>[9](#imagenet)</sup>)에서 자주 사용되는 Residual Network (ResNet)<sup>[10](#deep)</sup>을 baseline으로 고르게 되었습니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/BaselineResNet.png" alt="Baseline ResNet architecture" style="width:50%; height:50%"/>
    <figcaption><font size=2pt>
    	저희가 정의한 search space에서 표현한 ResNet block
  	 </font></figcaption>
  </center>
</figure>

ResNet의 여러 버전 중에서도 저희는 가장 모수가 적은 모델인 ResNet18을 활용하였습니다. 위 그림은 ResNet18에서의 한 block을 저희가 정의한 search space를 이용해 표현한 것인데요, skip connection이 위 그림에서 우측에 있는 $1 \times 1$ separable convolution과 identity로 표현되는 것을 확인할 수 있습니다. 저희는 이러한 block을 4개 쌓아서 전체 신경망 구조를 만들어주었습니다. 이러한 구조로 MNIST 데이터를 학습한 결과 98.94%의 정확도를 얻을 수 있었습니다.

#### SST-2 Dataset

한편 SST-2에서의 SOTA는 Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer<sup>[11](#limits)</sup>라는 논문의 알고리즘이었는데요, 무려 97.4%의 정확도를 보여주었습니다. 저희는 이 논문에서 활용한 Transformer<sup>[12](#attention)</sup> 구조를 baseline으로 삼았는데요, 성능을 올리기 위해 여러가지 시도를 하였으나 상대적으로 저조한 78.17%의 정확도까지밖에 달성하지 못했습니다. 이는 해당 논문에서 SST-2 뿐 아니라 웹으로부터 크롤링한 여러 데이터를 함께 사용하였고, 다른 데이터로부터 학습한 내용을 이전 (transfer)하는 기술을 활용한데서 오는 차이로 생각됩니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/BaselineTransformer.png" alt="Baseline Transformer architecure" style="width:50%; height:50%"/>
    <figcaption><font size=2pt>
    	저희가 정의한 search space에서 표현한 Transformer encoder block
    </font></figcaption>
  </center>
</figure>

Transformer에서의 self-attention layer를 구현해본 결과입니다. 각 층에 두개씩의 노드가 있어 gradient가 여러 방향으로 흐름을 알 수 있습니다. Transformer에서와 같이 모든 층에서 identity가 선택될 경우의 수가 존재하여 이후의 layer로 곧바로 이어지는 skip connection이 존재합니다.

### NEAT Results

#### MNIST Dataset

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/NEATResNEt.png" alt="NEAT로 적합한 ResNet18" style="width:25%; height:25%"/>
    <figcaption><font size=2pt>
    	MNIST 데이터셋에서 학습한 NEAT 알고리즘
	</font></figcaption>
  </center>
</figure>

MNIST 데이터셋에서 학습한 NEAT 알고리즘은 baseline보다 높은 99.14%의 성능을 나타냈습니다. 그리 의미있어 보이지 않는 identity 항이 드러난다는 것이 특징입니다. NEAT 알고리즘에서 진화를 하는 과정에서 우연히 identity 항을 가지는 node gene이 발현되었고, 이것이 성능에 악영향을 미치지 않아 세대를 거듭하면서도 계속 살아남은 것으로 보입니다.

#### SST-2 Dataset

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/NEATtransformer.png" alt="NEAT로 적합한 Transformer" style="width:25%; height:25%"/>
    <figcaption><font size=2pt>
    	SST-2 데이터셋에서 학습한 NEAT 알고리즘
    </font></figcaption>
  </center>
</figure>

SST-2에서 학습한 NEAT 알고리즘은 baseline보다 낮은 75.57%의 성능을 보였는데요, 위 결과물에서 알 수 있듯 첫 node로부터 변이 및 교차를 통해 더욱 커다란 신경망 구조를 생성하는데 실패한 것으로 보입니다. 이는 node가 한 개일 경우의 성능이 두 개일 경우보다 우월할 때 나타날 수 있는 현상으로, 만약 신경망 구조가 점점 커짐에 따라 그 성능이 단조적으로 증가하지 않는다면 NEAT 알고리즘은 학습에 어려움을 겪을 수 있다는 점을 보여줍니다.

### DARTS Results

#### MNIST Dataset

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/DARTSResNet.png" alt="DARTS로 적합한 ResNet18" style="width:50%; height:50%"/>
    <figcaption><font size=2pt>
    	MNIST 데이터셋에서 학습한 DARTS 알고리즘
    	</font></figcaption>
  </center>
</figure>

MNIST에서 구현한 DARTS 알고리즘은 baseline보다 높은 99.13%의 성능을 보였습니다. 위 그림은 DARTS에서의 cell 하나를 표현한 것으로 전체 네트워크는 이를 4개 쌓은 형태입니다. MNIST가 단순한 데이터셋으로 알려져 있는 만큼 이를 풀기 위한 신경망 구조도 단순해진 것으로 생각됩니다.

#### SST-2 Dataset

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/DARTSTransformer.png" alt="DARTS로 적합한 Transformer" style="width:70%; height:70%"/>
    <figcaption><font size=2pt>
    	SST-2 데이터셋에서 학습한 DARTS 알고리즘
    </font></figcaption>
  </center>
</figure>

SST-2 데이터셋에서 구현한 DARTS 알고리즘은 baseline보다 높은 78.86%의 정확도를 나타냈습니다. 학습 과정에서 Transformer 구조에서의 self-attention layer와 같이 multi-head attention과 identity로 이루어진 skip connection이 우측에 형성되었음을 확인할 수 있습니다.

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/MNIST_c.png" alt="MNIST에서의 3가지 신경망 구조 비교" style="width:80%; height:80%"/>
    <figcaption><font size=2pt>
    	MNIST 데이터셋의 baseline과 NEAT, DARTS 신경망 구조 비교
    </font></figcaption>
  </center>
</figure>

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/SST2_c.png" alt="SST-2에서의 3가지 신경망 구조 비교" style="width:80%; height:80%"/>
    <figcaption><font size=2pt>
    	SST-2 데이터셋의 baseline과 NEAT, DARTS 신경망 구조 비교
    </font></figcaption>
  </center>
</figure>

### Transfer Results

<figure>
  <center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_2019autoML/Result.png" alt="Experiment results"/>
    <figcaption><font size=2pt>
    	4가지 데이터셋에서의 baseline과 NEAT, DARTS의 성능 비교. <br> MNIST를 이용해 학습한 신경망 구조는 CIFAR10 데이터셋에서, SST-2는 IMDb 데이터셋에서 측정
    </font></figcaption>
  </center>
</figure>

저희는 앞서 MNIST, SST-2로부터 생성한 신경망 구조를 이용해 각각 다른 이미지 데이터셋 (CIFAR10<sup>[8](#cifar)</sup>)과 자연어처리 데이터셋 (IMDb<sup>[13](#learning)</sup>)에 학습시켜보고 그 결과를 비교해보았습니다.

우선 MNIST에서는 NEAT와 DARTS에서 탐색한 구조가 baseline model보다 더 좋은 성능을 보인 반면 CIFAR10에서는 baseline model이 더 우월한 성능을 보였습니다. 이는 두 알고리즘이 MNIST에서 생성한 구조가 baseline model에 비해 훨씬 단순하다는 점을 미루어볼 때, MNIST보다 크기가 크고 복잡한 이미지 데이터인 CIFAR10에서 해당 신경망 구조들이 과소적합 (underfitting) 되었기 때문으로 보입니다. NEAT의 경우 entire structure search를 목표로 하므로 이렇게 작은 데이터셋에서 큰 데이터셋으로 학습 결과를 이전 (transfer)하는데 어려움이 있지만, DARTS의 경우 생성한 cell를 여러 층 더 쌓게 되면 과소적합 문제를 해결할 수 있을 것으로 생각됩니다.

한편 baseline model과 NEAT, DARTS에서 만든 model 모두 SST-2에 비해 IMDb에서는 성능이 더욱 향상되었습니다. 매우 단순한 구조였던 NEAT에서의 신경망에서도 성능이 증가한 점을 볼 때, 특별한 전처리를 하지 않는 경우 IMDb에서의 분류 문제가 SST-2보다 쉽다는 것을 알 수 있습니다. SOTA 결과들을 보면 SST-2에서의 성능이 IMDb보다 일반적으로 높은 점을 생각하면 부자연스러운 결과인데요, SST-2에서는 전체 문장 뿐 아니라 문장에 포함된 어구들도 제공해주는데 저희가 이를 활용하지 않았기 때문으로 추측합니다. 각 데이터의 전처리 과정은 [DIYA 동아리 저장소](https://gitlab.diyaml.com/automl/automl-search)에 공개되어 있습니다.

따라서 SST-2 $\rightarrow$ IMDb의 경우 MNIST $\rightarrow$ CIFAR10과 반대로 복잡한 데이터셋에서 단순한 데이터셋으로 신경망 구조를 이전하는 것으로 볼 수 있습니다. 비록 NEAT는 복잡한 신경망 구조를 학습하는데는 실패했지만, NEAT와 DARTS 모두 이 경우 성능이 더욱 향상되었습니다. DARTS는 baseline model보다 node의 수를 적게 사용하면서도 두 데이터셋에서 모두 더욱 뛰어난 성능을 보여주었습니다.

## <a name="concl">5. Conclusion</a>

NAS의 다양한 유형에 대해 알아보고, 이 중 진화 알고리즘의 일종인 NEAT와 경사하강법을 이용한 탐색 방법의 일종인 DARTS를 저희가 직접 구현해본 결과에 대해 함께 살펴보았는데요, 우선 두 알고리즘이 생성하는 신경망 구조에는 큰 차이가 있었습니다. NEAT는 작고 단순한 구조에서부터 크고 복잡한 구조로, DARTS는 모든 연산자를 다 넣은 복잡한 구조에서부터 단순한 구조로 학습하는 방법이기 때문입니다. 이는 NEAT가 attention, skip connection과 같이 복잡한 연산을 필요로 하는 SST-2 데이터셋에서 학습에 실패한 이유이기도 할 것입니다. 이와 달리 DARTS는 SST-2에서는 baseline model과 같이 복잡한 신경망 구조를 만들었고, MNIST에서는 매우 단순한 구조를 만들 수 있었죠.

그러나 DARTS가 MNIST에서 해당 구조를 찾는데까지는 NEAT에 비해 상대적으로 오랜 시간이 걸렸습니다. 이번 실험은 이미지와 자연어처리 모두 비교적 단순한 데이터들을 사용했기에 DARTS가 대부분의 경우에서 NEAT보다 뛰어난 성능을 보였지만, 만약 경사하강법을 통한 구조의 학습이 쉽지 않은 경우 (gradient vanishing, dead relu 등)나 search space가 다계층으로 이루어져있는 경우 (multi-level optimization을 해야하는 경우)였다면 동일한 시간과 자원을 이용해 NEAT보다 좋은 성능을 내기가 어려웠을 수도 있습니다. 따라서 두 알고리즘의 성능은 그 성격이 데이터셋과 얼마나 부합하는지에 따라 결정된다고 보아야할 것입니다.

최근 NAS의 추세는 위 2가지 알고리즘이 아닌, 강화학습을 이용한 알고리즘들이라고 합니다.<sup>[1](#survey)</sup> 아쉽게도 이번 포스트에서는 해당 알고리즘을 직접 구현해보지 못했는데요, DIYA에서는 앞으로도 NAS의 여러가지 최신 기법을 소개하고 구현 과정을 공유하고자 하니 다음 블로그 포스트를 기대해주세요!


## Reference

<!-- 1 -->
##### <sup><a name="survey"></a>1</sup><sub><a href="https://arxiv.org/pdf/1908.00709.pdf" target="_blank">AutoML: A Survey of the State-of-the-Art, He et al, arXiv preprint, 2019.</a></sub>
<!-- 2 -->
##### <sup><a name="evolving"></a>2</sup><sub><a href="https://arxiv.org/pdf/1908.00709.pdf" target="_blank">Evolving Neural Networks through Augmenting Topologies, Stanley & Miikkulainen, Evolutionary computation 10(2), 2002.</a></sub>
<!-- 3 -->
##### <sup><a name="darts"></a>3</sup><sub><a href="https://arxiv.org/pdf/1908.00709.pdf" target="_blank">DARTS: Differentiable Architecture Search, Liu et al, arXiv preprint, 2019.</a></sub>
<!-- 4 -->
##### <sup><a name="html"></a>4</sup><sub><a href="http://khanrc.github.io/nas-2-darts-explain.html" target="_blank">http://khanrc.github.io/nas-2-darts-explain.html</a>
<!-- 5 -->
##### <sup><a name="MNIST"></a>5</sup><sub><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MNIST-SPM2012.pdf" target="_blank">The MNIST Database of Handwritten Digit Images for Machine Learning Research, Deng, L., IEEE Signal Processing Magazine 29(6), 2012.</a></sub>
<!-- 6 -->
##### <sup><a name="nlp"></a>6</sup><sub><a href="https://nlp.stanford.edu/sentiment" target="_blank">https://nlp.stanford.edu/sentiment</a></sub>
<!-- 7 -->
##### <sup><a name="code"></a>7</sup><sub><a href="http://paperswithcode.com" target="_blank">http://paperswithcode.com</a>
<!-- 8 -->
##### <sup><a name="cifar"></a>8</sup><sub><a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">https://www.cs.toronto.edu/~kriz/cifar.html</a></sub>
<!-- 9 -->
##### <sup><a name="imagenet"></a>9</sup><sub><a href="https://www.researchgate.net/profile/Li_Jia_Li/publication/221361415_ImageNet_a_Large-Scale_Hierarchical_Image_Database/links/00b495388120dbc339000000/ImageNet-a-Large-Scale-Hierarchical-Image-Database.pdf" target="_blank">Imagenet: A large-scale hierarchical image database., Deng et al, CVPR, 2009.</a></sub>
<!-- 10 -->
##### <sup><a name="deep"></a>10</sup><sub><a href="https://arxiv.org/pdf/1512.03385.pdf" target="_blank">Deep Residual Learning for Image Recognition, He et al, CVPR, 2016.</a></sub>
<!-- 11 -->
##### <sup><a name="limits"></a>11</sup><sub><a href="https://arxiv.org/pdf/1910.10683.pdf" target="_blank">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, Raffel et al, arXiv preprint, 2019.</a></sub>
<!-- 12 -->
##### <sup><a name="attention"></a>12</sup><sub><a href="http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf" target="_blank">Attention Is All You Need, Vaswani et al, NeurIPS, 2017.</a></sub>
<!-- 13 -->
##### <sup><a name="learning"></a>13</sup><sub><a href="https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf" target="_blank">Learning Word Vectors for Sentiment Analysis, Maas et al, ACL, 2011.</a></sub>

-----------

 Written by 
  김대영 | 김영훈 | 배수한 | 엄덕준 | 이재성