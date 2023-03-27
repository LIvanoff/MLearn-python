from arl.apriori import Apriori
from clustering.KMeans import KMeans
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from regression.Linear import Linear
from bayes.bayesian_classifier import Bayes

start = time.time()
if __name__ == "__main__":
    # arc = Apriori(filename='data.json', min_length=2)
    # arc.apriori(excel=True)

    # X1 = np.random.normal(loc=[0, -10], size=(100, 2))
    # X2 = np.random.normal(loc=[-10, 0], size=(100, 2))
    # X3 = np.random.normal(loc=[0, 0], size=(100, 2))
    # X = np.vstack((X1, X2, X3))

    # km = KMeans(10)
    # markers = km.fit_predict(X_old)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # axs[1].scatter(X[:, 0], X[:, 1], c=markers)
    # axs[0].scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()
    # df_train = pd.read_excel('test.xlsx', engine='openpyxl')

    # X = df_train['square'].to_numpy()
    # Y = df_train['clusters'].to_numpy()
    # X = np.array([1, 1.2, 1.6, 1.78, 2, 2.3, 2.4, 3, 3.3, 4, 4.1, 4.12, 4.34, 5, 5.3, 5.6, 6])
    # Y = np.array([0.8, 1, 0.9, 1.0, 1.2, 1.1, 1.6, 1.7, 2.0, 2.1, 2.15, 2.22, 2.45, 2.6, 2.12, 2.45, 2.3])
    #
    # regression = Linear(learning_rate=0.01,
    #                     max_iter=100,
    #                     optimizer_name='SGD')
    #
    # regression.fit(X=X, Y=Y)
    # x_loss1 = np.arange(len(regression.loss_history))
    #
    # fig1, axe = plt.subplots(1, 2, figsize=(15, 6))
    # axe[0].scatter(X, Y, marker='o', alpha=0.8)
    # axe[0].plot(X, regression.pred, 'r')
    # axe[1].plot(x_loss1, regression.loss_history)
    # plt.show()
    # print(regression.r1_score())
    # print(regression.r2_score())
    # regression2 = Linear(learning_rate=0.1,
    #                     max_iter=100,
    #                     optimizer_name='Adam',
    #                     batch_size=13)
    #
    # regression2.fit(X=X, Y=Y)
    # x_loss2 = np.arange(len(regression2.loss_history))
    #
    # regression3 = Linear(learning_rate=0.01,
    #                     max_iter=100,
    #                     optimizer_name='RMSprop',
    #                     batch_size=13)
    #
    # regression3.fit(X=X, Y=Y)
    # x_loss3 = np.arange(len(regression3.loss_history))
    #
    # plt.plot(x_loss1, regression.loss_history, 'r', label='SGD')
    # plt.plot(x_loss2, regression2.loss_history, 'b', label='Adam')
    # plt.plot(x_loss3, regression3.loss_history, 'g', label='RMSprop')
    # plt.legend()
    # plt.show()

    test = pd.read_excel('bayes_test_decode.xlsx', engine='openpyxl')
    data = test.values
    y = data[:, 3:4]
    x = data[:, :3]
    bayes = Bayes(2)
    condition_array = np.array(['да', 'да', 'нет'])
    print(bayes.predict(X=x, Y=y, condition_array=condition_array))


end = time.time()
print(" \n", (end - start) * 10 ** 3, "ms")
