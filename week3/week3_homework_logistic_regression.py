# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:06:27 2019

@author: Admin
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def logi_regression(w, b, x):
    pred_y = 1 / (1 + np.e**(-w * x - b))
    return pred_y



def cost_logi_regression(w, b, x_list, y_list):
    np_x_list = np.array(x_list)
    np_y_list = np.array(y_list)
    single_j = (logi_regression(w, b, np_x_list) - np_y_list) * np_x_list
    j_logi_regression = np.average(single_j)
    
    return j_logi_regression

def gradient(w, b, x, y):
    dw = (logi_regression(w, b, x) - y) * x
    db = logi_regression(w, b, x) - y
    return dw, db

def cal_step_gradient(x_list, y_list, w, b, lr):
    np_x_list = np.array(x_list)
    np_y_list = np.array(y_list)
    dw_list, db_list = gradient(w, b, np_x_list, np_y_list)
    aver_dw = np.average(dw_list)
    aver_db = np.average(db_list)
    w -= lr * aver_dw
    b -= lr * aver_db
    return w, b

def train(x_list, fact_y_list, batch_size, lr, times):
    w = 0
    b = 0
    loss_list = []
    for k in range(times):
        choice_list = np.random.choice(len(x_list), batch_size)
        x_train_list = []
        y_train_list = []
        
        for i in range(len(choice_list)):
            x_train_list.append(x_list[choice_list[i]])
            y_train_list.append(fact_y_list[choice_list[i]])
        w, b = cal_step_gradient(x_train_list, y_train_list, w, b, lr)
        print('w:{0},b:{1}'.format(w,b))
        print('loss is {0}'.format(cost_logi_regression(w, b, x_list, fact_y_list)))
        loss_list.append(cost_logi_regression(w, b, x_list, fact_y_list))

        
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(1,times+1),loss_list)
    ax.set_ylim([0,max(loss_list)])
    ax.set_xlabel('times')
    ax.set_ylabel('loss')
    ax.set_title('Train loss')
    plt.show()
    
    

    
def get_data():
    w = random.randint(0,10)+random.random()
    b = random.randint(0,5)+random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0,100)*random.randint(-1,1)
        y = logi_regression(w, b, x)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b

def run():
    x_list, y_list, w, b = get_data()
    lr = 0.001
    times = 1000
    train(x_list, y_list, 50, lr, times)
    
if __name__ == '__main__':
    run()








    