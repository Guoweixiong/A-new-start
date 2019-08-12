# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:19:37 2019

@author: Admin
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def get_y (w, b, x):
    pred_y = w * x + b
    return pred_y

def cost_func(w, b, x_list, y_list):
    x_np_list = np.array(x_list)
    y_np_list = np.array(y_list)
    #len_x = len(x_list)
    
    single_j_cost = (get_y(w, b, x_np_list) - y_np_list)**2
    j_cost = 0.5 * np.average(single_j_cost) #np.float64
    return j_cost

def gradient(pred_y, fact_y, x):
    diff = pred_y - fact_y
    dw = diff * x
    db = diff
    return dw, db

def cal_step_gradient(x_list, y_list, w, b, lr):
    fact_y_np = np.array(y_list)
    x_np = np.array(x_list)
    pred_y_np = get_y(w, b, x_np)
    dw_np, db_np = gradient(pred_y_np, fact_y_np, x_np)
    aver_dw = np.average(dw_np) #np.float64
    aver_db = np.average(db_np) #np.float64
    w = - lr * aver_dw #np.float64
    b = - lr * aver_db #np.float64
    return w, b
def train(x_list, y_list, lr, batch_size, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    loss_list = []
    for i in range(max_iter):
        train_samples = np.random.choice(num_samples,batch_size)
        train_sample_x = [x_list[j] for j in train_samples]
        train_sample_y = [y_list[j] for j in train_samples]
        w,b = cal_step_gradient(train_sample_x, train_sample_y, w, b, lr)
        print('w:{0},b:{1}'.format(w,b))
        print('loss is {0}'.format(cost_func(w, b, x_list, y_list)))
        loss_list.append(cost_func(w, b, x_list, y_list))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(1,max_iter+1),loss_list)
    ax.set_ylim([0,max(loss_list)])
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Train loss')
    plt.show()
    

      
def get_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0,5) + random.random()
    samples_num = 100
    x_list = []
    y_list = []
    for i in range(samples_num):
        x = random.randint(0,100)*random.random()
        y = w * x + b
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b

def run():
    x_list, y_list, w, b = get_data()
    lr = 0.00095
    max_iter = 1000
    train(x_list, y_list, lr, 50, max_iter)
    
if __name__ == '__main__':
    run()
    
    



    
    
    

    
    