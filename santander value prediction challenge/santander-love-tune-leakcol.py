import lightgbm as lgb
from sklearn import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# # from top scoring kernels and blends - for testing only
# sub1 = pd.read_csv(SCRIPT_PATH + '/Lovedataset/sub1.csv').rename(index=str, columns={"target": "target1"})
# sub2 = pd.read_csv(SCRIPT_PATH + '/Lovedataset/sub2.csv').rename(index=str, columns={"target": "target2"})
sub3 = pd.read_csv(SCRIPT_PATH + '/Lovedataset/sub3.csv')


# # standard
train_n = pd.read_csv(SCRIPT_PATH +'/train.csv')
test_n = pd.read_csv(SCRIPT_PATH +'/test.csv')

print('data load complete',train_n.shape)

def tuneleak(leak_col,train,test):
    col = list(leak_col)
    train = train[col +  ['ID', 'target']]
    test = test[col +  ['ID']]
    train.loc[:,"nz_mean"] = train[col].apply(lambda x: x[x!=0].mean(), axis=1)
    train.loc[:,"nz_max"] = train[col].apply(lambda x: x[x!=0].max(), axis=1)
    train.loc[:,"nz_min"] = train[col].apply(lambda x: x[x!=0].min(), axis=1)
    train.loc[:,"ez"] = train[col].apply(lambda x: len(x[x==0]), axis=1)
    train.loc[:,"mean"] = train[col].apply(lambda x: x.mean(), axis=1)
    train.loc[:,"max"] = train[col].apply(lambda x: x.max(), axis=1)
    train.loc[:,"min"] = train[col].apply(lambda x: x.min(), axis=1)
    test.loc[:,"nz_mean"] = test[col].apply(lambda x: x[x!=0].mean(), axis=1)
    test.loc[:,"nz_max"] = test[col].apply(lambda x: x[x!=0].max(), axis=1)
    test.loc[:,"nz_min"] = test[col].apply(lambda x: x[x!=0].min(), axis=1)
    test.loc[:,"ez"] = test[col].apply(lambda x: len(x[x==0]), axis=1)
    test.loc[:,"mean"] = test[col].apply(lambda x: x.mean(), axis=1)
    test.loc[:,"max"] = test[col].apply(lambda x: x.max(), axis=1)
    test.loc[:,"min"] = test[col].apply(lambda x: x.min(), axis=1)
    col += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']

    for i in range(2, 100):
        train.loc[:,'index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
        test.loc[:,'index'+str(i)] = ((test.index + 2) % i == 0).astype(int)
        col.append('index'+str(i))

    test = pd.merge(test, sub3, how='left', on='ID',)
    from scipy.sparse import csr_matrix, vstack
    # train = train.replace(0, np.nan)
    # test = test.replace(0, np.nan)
    train = pd.concat((train, test), axis=0, ignore_index=True)

    # folds = 1
    # for fold in range(folds):
    #     x1, x2, y1, y2 = train_test_split(train[col], np.log1p(train.target.values), test_size=0.20, random_state=fold)
    #     print('x1 = ',x1)
    #     print('x2 = ',x2)
    #     print('y1 = ',y1)
    #     print('y2 = ',y2)

    #     model = lgb.train(params, lgb.Dataset(x1, label=y1), 3000 ,lgb.Dataset(x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    
    #     test['target'] += np.expm1(model.predict(test[col], num_iteration=model.best_iteration))
    # test['target'] /= folds
    # test[['ID', 'target']].to_csv('submission-love.csv', index=False)

    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    from sklearn.metrics import mean_squared_error 
    train_t = train[col]
    y_train = np.log1p(train.target.values)
    NUM_FOLDS = 5 #need tuned
    def rmsle_cv(model):
        kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train_t.values)
        rmse= np.sqrt(-cross_val_score(model, train_t, y_train, scoring="neg_mean_squared_error", cv = kf))
        return(rmse)

    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=60,
                              learning_rate=0.01, max_depth=7,
                              metric='rmse',is_training_metric=True,
                              bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9) 

    score = rmsle_cv(model_lgb)
    return score

bestleak1 = 0
bestleak2 = 0
minscore = 0
col = [c for c in train_n.columns if c not in ['ID', 'target']]
for i_1 in range(30,80,10):
    for i_2 in range(3000,4500,500):
        leak_col = []
        for c in col:
            leak1 = np.sum((train_n[c]==train_n['target']).astype(int))
            leak2 = np.sum((((train_n[c] - train_n['target']) / train_n['target']) < 0.05).astype(int))
            if leak1 > i_1 and leak2 > i_2 :
                leak_col.append(c)
        print("l1 = "+ str(i_1) +', l2 = ' + str(i_2), ' len = ' ,len(leak_col))

        score = tuneleak(leak_col,train_n.copy(),test_n.copy())
        print("l1 = "+ str(i_1) +', l2 = ' + str(i_2))
        print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

        with open(SCRIPT_PATH + "/Lovedataset/love-tube-col-l1"+ str(i_1) +'-l2-' + str(i_2) + ".txt", 'w') as f:
                for s in col: f.write(str(s))
                f.write("l1 = " + str(i_1) + ', l2 = ' + str(i_2) + ' len = ' + str(len(leak_col)) + '\n')
                f.write("LGBM score mean =  " + str(score.mean()) + " std = " + str(score.std()) + '\n')
        
        if  score.mean() > minscore:
            minscore = score.mean()
            bestleak1 = i_1
            bestleak2 = i_2
            print('bestleak1 = ', bestleak1, 'bestleak2 = ', bestleak2)

print('bestleak1 = ', bestleak1, 'bestleak2 = ', bestleak2)
            
# col = []
# f=open(SCRIPT_PATH + "/Lovedataset/submission-love-col.txt",'r')
# lines=f.readlines()
# for line in lines:
#     col.append(line[:-1])
