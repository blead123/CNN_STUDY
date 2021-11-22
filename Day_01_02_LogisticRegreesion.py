import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



# 딥러닝으로 할 수 있는것 -->회귀 , 분류
def logistic_regression():
    ## [공부시간 , 출석일수]
    x = [[1, 2], [2, 3], [4, 5], [5, 4], [8, 9], [9, 8]]
    ## 0->탈락 1->통과
    y = [[0], [0], [1], [1], [1], [1]]
    model = keras.models.Sequential()
    ###로지스틱이랑 멀티플이랑 다름 -->cnn에선 멀티플 안씀
    model.add(keras.layers.Dense(1, keras.activations.sigmoid))  ##모든 출력의 결과가 0~1사이 w=x+b
    model.compile(optimizer=keras.optimizers.SGD(0.1),  ##옵티마이저가 값로스를 줄이면서 최적화에 가까워짐
                  loss=keras.losses.binary_crossentropy, metrics='acc')  ##로그 방정식을 이용해 손실을 계산
    model.fit(x, y, epochs=20, verbose=2)
    p = model.predict(x)
    print(p)

    p_bool = (p > 0.5)
    print(p_bool)

    equels = (p == p_bool)
    print(equels)

    print('acc : ', np.mean(equels))


def logistic_regression_pima():
    pima = pd.read_csv('data/pima-indians-diabetes.csv', skiprows=9, header=None)  ##skiporw==>줄을 아홉번 뛰어넘음 헤더 없앰

    print(pima)

    print(pima.values)

    x = pima.values[:, :-1]
    y = pima.values[:, -1:]

    x=preprocessing.scale(x) # 데이터 전처리 평균과 표준편차 이용-->표준화
    #x=preprocessing.minmax_scale()#0,1사용-->둘중 하나 사용-->일반화

    ##사이킷 런으로 나눌수 있음음
    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # print(x_train.shape, x_test.shape)    # (537, 8) (231, 8)
    # print(y_train.shape, y_test.shape)    # (537, 1) (231, 1)

    print(x_train.shape, x_test.shape)




    model=keras.Sequential()
    model.add(keras.layers.Dense(1,activation=keras.activations.sigmoid))
    model.compile(optimizer=keras.optimizers.SGD(0.01),loss=keras.losses.binary_crossentropy,metrics='acc')
    model.fit(x_train,y_train ,epochs=200, verbose=2 ,batch_size=64)

    p=model.predict(x_test)
    p_bool=(p>0.5)

    print('acc :' , np.mean(p_bool==y_test))


logistic_regression_pima()
