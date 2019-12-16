# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:35:05 2019

@author: Admin
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random

clf = LinearRegression()
tran = PolynomialFeatures()  # 用多项式生成多个特征

def ransac(data,model,epochs,n,t,d):
    '''
    data,数据集，二维数组[X,y]
    epochs,迭代次数
    n,拟合模型所用数据数量，即随机抽取得到样本数目
    t,判定是否是内点的阈值
    model，用于拟合的模型
    d,判定模型为好模型的内点数目
    '''
    best_model = None
    best_err = np.inf
    best_inlier = None
    best_inlier_idx = None
    
    epoch = 0
    while epoch < epochs:
        # 随机抽取n个样本，用于拟合model
        train_idx,test_idx = divide(data,n)
        #inlier = [data[i] for i in train_idx] # 内点集合
        inlier = data[train_idx,:]
        may_inlier = data[test_idx, :]
       
#        test_inlier_idx = [] # 判定为内点的索引的集合

        
        may_model = clf.fit(tran.fit_transform(inlier[:,:-1]), inlier[:,-1])

        test_error,_ = get_error(may_model, may_inlier[:,:-1], may_inlier[:,-1])

        also_idx = [i for i in range(len(test_error)) if test_error[i] < t]

        print(also_idx)
        print(type(also_idx))
        also_inlier = data[also_idx,:]
        print(type(also_inlier))
        if len(also_inlier) > d:
            better_data = np.concatenate((inlier, also_inlier))

            better_model = clf.fit(tran.fit_transform(better_data[:,:-1]), better_data[:,-1])
            _,this_error=get_error(better_model,better_data[:,:-1], better_data[:,-1])
            
            print(this_error)
            
            if this_error < best_err:
                best_model = better_model
                best_err = this_error
                best_inlier = better_data
                best_inlier_idx = train_idx
        # circle
        epoch += 1
    if best_model == None:
        raise ValueError("无法拟合出model")
    else:
        return best_model, best_err, best_inlier_idx, best_inlier, best_model.coef_ # 这里参数有十个，是因为[x0,x1,x2]后面使用PolynomialFeatures进行了特征处理，变成了十维

       

def divide(data, n):
    dataset_num = len(data)
    idxs = np.arange(dataset_num)
    np.random.shuffle(idxs)
    return idxs[:n], idxs[n:]

def get_error(model,test,true_y):
    pred_y = model.predict(tran.fit_transform(test))
    error = np.sqrt((pred_y - true_y)**2)
    mean_error = np.mean(error)
    return error, mean_error

# ----------------------------------------------------------------------------------------
model = LinearRegression


def get_data():
    w0 = random.randint(0,10) + random.random()
    w1 = random.randint(0,10) + random.random()
    w2 = random.randint(10,30) + random.random()
    w = [w0, w1, w2]
    w = np.array(w)
    data = []
    
    #print(type(data))
    for i in range(150):
        x0 = random.randint(1,100)+random.random()
        x1 = random.randint(3,80)+random.random()
        x2 = random.randint(1,9)+random.random()
        x = np.array([x0, x1, x2])
        y = int(np.dot(w, x))

        x_y = np.append(x, y)
        data.append(x_y)
    
    v0 = w0 * 1.1
    v1 = w1 * 1.4
    v2 = w2 * 1.2
    v = [v0, v1, v2]
    v = np.array(v)
    for k in range(10):
        x0 = random.randint(1,100)+random.random()
        x1 = random.randint(3,80)+random.random()
        x2 = random.randint(1,9)+random.random()
        x = np.array([x0, x1, x2])
        y = int(np.dot(v, x))

        x_y_v = np.append(x, y) 
        data.append(x_y_v)
    data = np.array(data)
    return data
 
model = LinearRegression()
epochs = 50
n = 60
t = 5
d = 20
data = get_data()
print(ransac(data,model,epochs,n,t,d))
if __name__ == '__main__':
    print("good")
'''
data,数据集，二维数组[X,y]
epochs,迭代次数
n,拟合模型所用数据数量，即随机抽取得到样本数目
t,判定是否是内点的阈值
model，用于拟合的模型
d,判定模型为好模型的内点数目
'''


        
    
    




