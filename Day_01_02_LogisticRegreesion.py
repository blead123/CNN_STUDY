import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn import preprocessing

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
    pima = pd.read_csv('data/pima-indians-diabetes.csv', skiprows=9, header=None)  ##skiporw==>줄을 아홉번 뛰어넘음
    print('pima')
    print(pima)
    print('pima val')
    print(pima.values)
    print('x value')
    x = pima.values[:, :-1]
    print(x)
    print('y value')
    y = pima.values[:, -1:]
    print(y)
    x=preprocessing.scale(x) # 데이터 전처리
    model=keras.Sequential()
    model.add(keras.layers.Dense(1,activation=keras.activations.sigmoid))
    model.compile(optimizer=keras.optimizers.SGD(0.01),loss=keras.losses.binary_crossentropy,metrics='acc')
    model.fit(x,y ,epochs=200, verbose=2)

    p=model.predict(x)
    p_bool=(p>0.5)

    print('acc :' , np.mean(p_bool==y))


logistic_regression_pima()
