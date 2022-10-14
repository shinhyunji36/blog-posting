# Gradcient_Descent(경사 하강법)

- feature가 2개라고 가정: `f1`, `f2`

## weight와 bias를 업데이트 하는 함수
```python
def get_update_weights_value(bias, w1, w2, f1, f2, target, learning_rate=0.01):

    # 데이터 개수
    N = len(target)
    # 예측값
    predicted = w1*f1 + w2*f2 + bias
    # 실제값 - 예측값 (차이)
    diff = target - predicted
    # bias를 array 기반으로 구하기 위해 설정
    bias_factors = np.ones((N,)) # bias도 weight처럼 update 해주려 생성

    # weight와 bias를 얼마나 update할 것인지를 계산
    w1_update = -(2/N)*learning_rate*(np.dot(f1.T, diff))
    w2_update = -(2/N)*learning_rate*(np.dot(f2.T, diff))
    bias_update = -(2/N)*learning_rate(np.dot(bias_factors.T, diff))

    # Mean Sqaured Error 값 계산
    mse_loss = np.mean(np.square(diff))

    # weight와 bias가 update되어야 할 값과 Mean Squared Error 값을 반환
    return bias_update, w1_update, w2_update, mse_loss
```

### gradient descent를 적용하는 함수
```python
def gradient_descent(features, target, iter_epochs=100, verbos=True):
    # w1, w2 numpy array 연산을 위해 1차원 array로 변환하되 초기 값은 0으로 설정
    # bias도 1차원 array로 변환하되, 초기 값은 1로 설정 
    w1 = np.zeros((1,)) # array([0. ])
    w2 = np.zeros((1,))
    bias = np.ones((1,)) # array([1. ])
    print('최초 w1, w2, bias:', w1, w2, bias)

    # learning_rate와 f1, f2 피쳐 지정. 호출 시  numpy array형태로 f1와 f2로 된 2차원 feature가 입력된다.
    learning_rate = 0.01
    f1 = features[:, 0]
    f2 = features[:, 1]

    # iter_epochs 수만큼 반복하면서 weight와 bias update 수행
    for i in range(iter_epochs):
        # weight/bias update 값 계산
        bais_update, w1_update, w2_update, loss = get_update_weights_value(bias, w1, w2, f1, f2, target, learning_rate)
        # weight/bias의 update 적용
        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update
        if verbose:
            print('Epoch:', i+1,'/', iter_epochs )
            print('w1:', w1, 'w2:', w2, 'bias:', bias, 'loss:', loss) # 여기서 loss는 loop를 돌면서 loss값이 주는 지 확인하는 용도

    return w1, w2, bias
```

## Stochastic Gradient Descent (SGD)
- SGD 기반으로 Weight/Bias update 값 구하기

```python
def get_update_weights_value_sgd(bias, w1, w2, f1, f2, target, learning_rate):

    # 데이터 개수
    N = target.shape[0]
    # 예측 값
    predicted = w1*f1 + w2*f2 + bias
    # 실제값 - 예측값 (차이)
    diff = target - predicted
    # bias를 array 기반으로 구하기 위해서 설정
    bias_factors = np.ones((N,))

    # weight와 bias를 얼마나 update 할 것인지 계산.
    w1_update = -(2/N)*learning_rate*(np.dot(f1.T, diff))
    w2_update = -(2/N)*learning_rate*(np.dot(f2.T, diff))
    bias_update = -(2/N)*learning_rate(np.dot(bias_factors.T, diff))
    
    return bias_update, w1_update, w2_update
```

## SDG gradient descent 적용
```python
def stochastic_gradient_descent(features, target, iter_epochs=1000, verbose=True):
    # w1, w2는 numpy array 연산을 위해 1차원 array로 변환하되 초기 값은 0으로 설정
    # bias도 1차원 array로 변환하되 초기 값은 1로 설정
    np.random.seeed = 2022
    w1 = np.zeros((1,))
    w2 = np.zeros((1,))
    bias = np.ones((1,))
    print('최초 w1, w2, bias:', w1, w2, bias)

    # learning_rate와 f1, f2 피처 지정. 호출 시 numpy array형태로 f1과 f2으로 된 2차원 feature가 입력됨.
    learning_rate = 0.01
    f1 = features[:, 0]
    f2 = features[:, 1]


    # iter_epochs 수만큼 반복하면서 weight와 bias update 수행
    for i in range(iter_epochs):
        # iteration 시마다 stochastic gradient descent를 수행할 데이터를 한개만 추출한다. 추출할 데이터의 인덱스를 random.choice()로 선택.
        stochastic_index = np.random.choice(target.shape[0], 1)
        f1_sgd = f1[stochastic_index]
        f2_sgd = f2[stochastic_index]
        target_sgd = target[stochastic_index]
        # SGD 기반으로 Weight/Bias의 update를 구함
        bias_update, w1_update, w2_update = get_update_weight_value_sgd(bias, w1, w2, f1_sgd, f2_sgd, target_sgd, learning_rate)

        # SGD로 구한 weight/bias의 update 적용
        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update
        if verbose:
            print('Epoch:', i+1,'/' iter_epochs)
            # Loss는 전체 학습 데이터 기반으로 구해야 함. 여기서 구한 mse_loss는 현재 epoch에서 선택된 데이터에 대한 loss 값
            predicted = w1*f1 + w2*f2 + bias
            diff = target - predicted
            mse_loss = np.mean(np.square(diff))
            print('w1:', w1, 'w2:', w2, 'bias:', bias, 'loss:', mse_loss)
    return w1, w2, bias
```

- mini-batch gradient descent는 하이퍼 파라미터인 batch_size를 받아서 그만큼의 batch data를 활용해 weight/bias를 update한다.

- SGD와 마찬가지로 Loss는 전체 학습 데이터 기반으로 구해야 한다.

    - iteration시 마다 일정한 batch 크기 만큼 데이터를 random하게 가져오는 mini-batch gradient descent

    ```python
    batch_indexes = np.random.choice(target.shape[0], batch_size)
    f1_batch = f1[batch_indexes]
    f2_batch = f2[batch_indexes]
    target_batch = target[batch_indexes]
    ```

    - iteration시 마다 일정한 batch 크기 만큼 데이터를 순차적으로 가져오는 mini-batch gradient descent
    
    ```python
    for batch_step in range(0, target.shape[0], batch_size):
        f1_batch = f1[batch_step:batch_step + batch_size]
        f2_batch = f2[batch_step:batch_step + batch_size]
        target_batch = target[batch_step:batch_step + batch_size]
    ```


- keras layer에서 kernel은 모두 weight
    - 따라서 `kernel_initialize='zeros'`는 weight 초기화
        - ex) gradient_descent()에서 `w1 = np.zeros((1, ))`을 사용해 0으로 초기화 했던 것과 동일
- `bias_initialize='ones'`는 bias 초기화
    - ex) gradient_descent()에서 `bias = np.ones((1,))` 을 사용해 1로 초기화 했던 것과 동일 