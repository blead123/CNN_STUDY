# Day_01_01_MultipleRegression.py
import tensorflow.keras as keras
import pandas as pd
import numpy as np

# 1. 파이참 프로젝트 생성
#    프로젝트 이름은 CNN
# 2. 프로젝트에 파일 추가
#    파일 이름은 Day_01_01_MultipleRegression.py

# 파일 실행 ctrl + shift + f10
# alt + 1
# alt + 4
# ctrl + c + v + v
# ctrl + x
# 주석 ctrl + /


def linear_regression():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))  # wx + b

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)  # mean square error

    model.fit(x, y, epochs=100, verbose=2)  # 0(없음) 1(전체) 2(약식)

    print(model.predict(x))

    # 퀴즈
    # x가 5와 7일 때의 결과를 예측하세요
    print(model.predict([5, 7]))


def multiple_regression():
    # 공부시간 출석일수
    x = [[1, 2],
         [2, 1],
         [4, 5],
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[3],
         [3],
         [9],
         [9],
         [17],
         [17]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    # inf(infinite), nan(not a number)
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)

    # 퀴즈
    # 3시간 공부하고 8번 출석한 학생과
    # 6시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(model.predict(x))
    print(model.predict([[1, 2],
                         [2, 1],
                         [4, 5],
                         [5, 4],
                         [8, 9],
                         [9, 8]]))
    print(model.predict([[3, 8],
                         [6, 2]]))


def multiple_regression_trees():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    print(trees)
    print('############################################')
    print(trees.values)
    print('############################################')
    x = trees.values[:, :-1]
    print('x value')
    print(x)
    print('############################################')
    y = trees.values[:, -1:]
    print('y value')
    print(y)
    print(x.shape, y.shape)         # (31, 2) (31, 1)

    # 공부시간 출석일수
    # x = [[1, 2],                  # (6, 2)
    #      [2, 1],
    #      [4, 5],
    #      [5, 4],
    #      [8, 9],
    #      [9, 8]]
    # y = [[3],                     # (6, 1)
    #      [3],
    #      [9],
    #      [9],
    #      [17],
    #      [17]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.0001),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)


# linear_regression()
# multiple_regression()
multiple_regression_trees()



