import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import math

class DTPLOT:
    def __init__(self, df_train):
        self.df_train = df_train
    def histogram(self):
        sns.distplot(self.df_train['SalePrice'])
    def scatterplot(self):
        var = 'GrLivArea'
        data = pd.concat([self.df_train['SalePrice'], self.df_train[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
        var = 'TotalBsmtSF'
        data = pd.concat([self.df_train['SalePrice'], self.df_train[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    def boxplot(self):
        var = 'OverallQual'
        data1 = pd.concat([self.df_train['SalePrice'], self.df_train[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data1)
        fig.axis(ymin=0, ymax=800000);
        var = 'YearBuilt'
        data2 = pd.concat([self.df_train['SalePrice'], self.df_train[var]], axis=1)
        f, ax = plt.subplots(figsize=(16, 8))
        fig = sns.boxplot(x=var, y="SalePrice", data=data2)
        fig.axis(ymin=0, ymax=800000);  
        plt.xticks(rotation=90);

    def heatmap(self):
        #correlation matrix
        corrmat = self.df_train.corr()
        f, ax = plt.subplots(figsize=(12, 9))
        hm1 = sns.heatmap(corrmat, vmax=.8, square=True)
        #saleprice correlation matrix
        k = 10 #number of variables for heatmap
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(self.df_train[cols].values.T)
        sns.set(font_scale=1.25)
        hm2 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

    def snssetscatterplot(self):
        #scatterplot
        sns.set()
        cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        sns.pairplot(self.df_train[cols], size = 2.5)
        plt.savefig(SCRIPT_PATH + "/myfig.png")
class DTPROCESS:
    def __init__(self, df_train):
        self.df_train = df_train
    def missdata(self):
        total = self.df_train.isnull().sum().sort_values(ascending=False)
        percent = (self.df_train.isnull().sum()/self.df_train.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        self.df_train = self.df_train.drop((missing_data[missing_data['Total'] > 4]).index,1)
        # print(missing_data.head(19))
        
        self.df_train["Functional"] = self.df_train["Functional"].fillna("Typ")
        self.df_train['MSZoning'] = self.df_train['MSZoning'].fillna(self.df_train['MSZoning'].mode()[0])
        self.df_train['Exterior1st'] = self.df_train['Exterior1st'].fillna(self.df_train['Exterior1st'].mode()[0])
        self.df_train['Exterior2nd'] = self.df_train['Exterior2nd'].fillna(self.df_train['Exterior2nd'].mode()[0])
        self.df_train['Electrical'] = self.df_train['Electrical'].fillna(self.df_train['Electrical'].mode()[0])
        self.df_train['SaleType'] = self.df_train['SaleType'].fillna(self.df_train['SaleType'].mode()[0])
        self.df_train['KitchenQual'] = self.df_train['KitchenQual'].fillna(self.df_train['KitchenQual'].mode()[0])
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            self.df_train[col] = self.df_train[col].fillna(0)
        for col in ('GarageArea', 'GarageCars'):
            self.df_train[col] = self.df_train[col].fillna(0)

        # print('\nmissingdata\n')    
        # print(self.df_train.isnull().sum().max())
        # print(missing_data.head(19))
    def trainoutliar(self):
        # main
        saleprice_scaled = StandardScaler().fit_transform(self.df_train['SalePrice'][:,np.newaxis]);
        low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
        high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
        # second
        # self.df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
        # self.df_train = self.df_train.drop(self.df_train[self.df_train['Id'] == 1299].index)
        # self.df_train = self.df_train.drop(self.df_train[self.df_train['Id'] == 524].index)
        self.df_train = self.df_train.drop(self.df_train[(self.df_train['GrLivArea']>4000) & (self.df_train['SalePrice']<300000)].index)
    def uselessdata(self):
        dropping = ['Id','Utilities']
        self.df_train.drop(dropping,axis=1, inplace=True)
    def categoricaltrchange(self):
        #MSSubClass=The building class
        self.df_train['MSSubClass'] = self.df_train['MSSubClass'].apply(str)
        #Changing OverallCond into a categorical variable
        self.df_train['OverallCond'] = self.df_train['OverallCond'].astype(str)
        #Year and month sold are transformed into categorical features.
        self.df_train['YrSold'] = self.df_train['YrSold'].astype(str)
        self.df_train['MoSold'] = self.df_train['MoSold'].astype(str)
    def normality(self):
        self.df_train['SalePrice'] = np.log(self.df_train['SalePrice'])
        # sns.distplot(self.df_train['SalePrice'], fit=norm);
        # fig = plt.figure()
        # res = stats.probplot(self.df_train['SalePrice'], plot=plt)
        self.df_train['GrLivArea'] = np.log(self.df_train['GrLivArea'])
        # sns.distplot(self.df_train['GrLivArea'], fit=norm);
        # fig = plt.figure()
        # res = stats.probplot(self.df_train['GrLivArea'], plot=plt)
        # totalbsmt log but have 0 
        # self.df_train['TotalBsmtSFlog'] = pd.Series(len(self.df_train['TotalBsmtSF']), index=self.df_train.index)
        # self.df_train['TotalBsmtSFlog'] = 0 
        # self.df_train.loc[self.df_train['TotalBsmtSF']>0,'TotalBsmtSFlog'] = 1
        # self.df_train.loc[self.df_train['TotalBsmtSFlog']==1,'TotalBsmtSFlog'] = np.log(self.df_train['TotalBsmtSF'])
        # sns.distplot(self.df_train[self.df_train['TotalBsmtSFlog']>0]['TotalBsmtSFlog'], fit=norm);
        # fig = plt.figure()
        # res = stats.probplot(self.df_train[self.df_train['TotalBsmtSFlog']>0]['TotalBsmtSFlog'], plot=plt)
    def LabelEncoder(self):
        from sklearn.preprocessing import LabelEncoder
        cols = ( 'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'Functional', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
        # process columns, apply LabelEncoder to categorical features
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(self.df_train[c].values)) 
            self.df_train[c] = lbl.transform(list(self.df_train[c].values))
    def skewedfeatures(self):
        from scipy.special import boxcox1p
        numeric_feats = self.df_train.dtypes[self.df_train.dtypes != "object"].index
        # Check the skew of all numerical features
        skewed_feats = self.df_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        # print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        # print(skewness.head(10))
        skewness = skewness[abs(skewness) > 0.75]
        # print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            #all_data[feat] += 1
            self.df_train[feat] = boxcox1p(self.df_train[feat], lam)
    def addfeatures(self):
        self.df_train['TotalSF'] = self.df_train['TotalBsmtSF'] + self.df_train['1stFlrSF'] + self.df_train['2ndFlrSF']
    def lookdummies(self):
        self.df_train = pd.get_dummies(self.df_train)
class ML:
    def __init__(self, train, test,y_train):
        self.train = train
        self.test = test
        self.y_train = y_train
    def startml(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score, KFold
        def modeling(clf,ft,target):
            acc = cross_val_score(clf,ft,target,cv=kf)
            acc_lst.append(acc.mean())
            return 
        def ml(ft,target,time):
            accuracy.append(acc_lst)
            #logisticregression
            logreg = LogisticRegression()
            modeling(logreg,ft,target)
            #RandomForest
            rf = RandomForestClassifier(n_estimators=50,min_samples_split=4,min_samples_leaf=2)
            modeling(rf,ft,target)
            #svc
            svc = SVC()
            modeling(svc,ft,target)
            #knn
            knn = KNeighborsClassifier(n_neighbors = 3)
            modeling(knn,ft,target)
            # see the coefficient
            logreg.fit(ft,target)
            feature = pd.DataFrame(ft.columns)
            feature.columns = ['Features']
            feature["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
            print(feature)
            return
        train_ft, test_ft, train_y, accuracy, testdata = self.train, self.test, self.y_train, [], []
        mldrop = [['SalePrice']]
        mltest = ['M1','M2','M3','M4','M5','M6']
        for dr in range(6):
            # train_ft = train.drop(mldrop[dr],axis=1)
            # test_ft = test.drop(mldrop[dr],axis=1)
            testdata.append([train_ft,test_ft])
            # kf = KFold(n_splits=3,random_state=1)
            kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
            acc_lst = []
            ml(testdata[dr][0],train_y,mltest[dr])
            # test_2 = test.drop('young',axis=1)

        accuracy_df=pd.DataFrame(data=accuracy,index=['M1','M2','M3','M4','M5','M6'],columns=['logistic','rf','svc','knn'])
        print (accuracy_df)
class ML2:
    def __init__(self, train, test,y_train):
        self.train = train
        self.test = test
        self.y_train = y_train
    def startml(self):
        from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.metrics import mean_squared_error
        import xgboost as xgb
        import lightgbm as lgb
        def cross(model):
            def rmsle_cv(model):
                kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
                rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
                print(cross_val_score(model,train.values,y_train,cv=kf).mean())
                print('acc no squrt')
                return(rmse)
            def rmsle(y, y_pred):
                return np.sqrt(mean_squared_error(y, y_pred))  
            lasso, ENet, KRR, GBoost, model_xgb, model_lgb, averaged_models = model[0], model[1], model[2], model[3], model[4], model[5], model[6]
            # score = rmsle_cv(lasso)
            # print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
            # score = rmsle_cv(ENet)
            # print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
            # score = rmsle_cv(KRR)
            # print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
            # score = rmsle_cv(GBoost)
            # print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
            # score = rmsle_cv(model_xgb)
            # print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
            # score = rmsle_cv(model_lgb)
            # print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std())) 
            # score = rmsle_cv(averaged_models)
            # print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
            stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
            score = rmsle_cv(stacked_averaged_models)
            print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))     
        
            stacked_averaged_models.fit(train.values, y_train)
            stacked_train_pred = stacked_averaged_models.predict(train.values)
            stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
            print(rmsle(y_train, stacked_train_pred))
            
            model_xgb.fit(train, y_train)
            xgb_train_pred = model_xgb.predict(train)
            xgb_pred = np.expm1(model_xgb.predict(test))
            print(rmsle(y_train, xgb_train_pred))
            
            model_lgb.fit(train, y_train)
            lgb_train_pred = model_lgb.predict(train)
            lgb_pred = np.expm1(model_lgb.predict(test.values))
            print(rmsle(y_train, lgb_train_pred))
            
            print('RMSLE score on train data:')
            print(rmsle(y_train,stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
            ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

            return ensemble

        class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
            def __init__(self, base_models, meta_model, n_folds=5):
                self.base_models = base_models
                self.meta_model = meta_model
                self.n_folds = n_folds
        
            # We again fit the data on clones of the original models
            def fit(self, X, y):
                self.base_models_ = [list() for x in self.base_models]
                self.meta_model_ = clone(self.meta_model)
                kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
                
                # Train cloned base models then create out-of-fold predictions
                # that are needed to train the cloned meta-model
                out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
                for i, model in enumerate(self.base_models):
                    for train_index, holdout_index in kfold.split(X, y):
                        instance = clone(model)
                        self.base_models_[i].append(instance)
                        instance.fit(X[train_index], y[train_index])
                        y_pred = instance.predict(X[holdout_index])
                        out_of_fold_predictions[holdout_index, i] = y_pred
                        
                # Now train the cloned  meta-model using the out-of-fold predictions as new feature
                self.meta_model_.fit(out_of_fold_predictions, y)
                return self
        
            #Do the predictions of all base models on the test data and use the averaged predictions as 
            #meta-features for the final prediction which is done by the meta-model
            def predict(self, X):
                meta_features = np.column_stack([
                    np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                    for base_models in self.base_models_ ])
                return self.meta_model_.predict(meta_features)
        class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
            def __init__(self, models):
                self.models = models
            # we define clones of the original models to fit the data in
            def fit(self, X, y):
                self.models_ = [clone(x) for x in self.models]
                # Train cloned base models
                for model in self.models_:
                    model.fit(X, y)
                return self
            #Now we do the predictions for cloned models and average them
            def predict(self, X):
                predictions = np.column_stack([
                    model.predict(X) for model in self.models_
                ])
                return np.mean(predictions, axis=1)   
        lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state =5)
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             seed =7, nthread = -1)
        model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
        model = [lasso,ENet,KRR,GBoost,model_xgb,model_lgb,averaged_models]
        return cross(model)
    
if __name__ == '__main__':
    SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
    df_train = pd.read_csv(SCRIPT_PATH + "/train.csv")
    df_test = pd.read_csv(SCRIPT_PATH + "/test.csv")
    df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
    ntrain, ntest, y_train, test_ID = df_train.shape[0], df_test.shape[0], df_train.SalePrice.values, df_test.Id.values
    y_train = np.log1p(y_train)
    # for x, value in np.ndenumerate(y_train):
    #     y_train[x] = long(y_train[x])
    # print("The train data size before dropping Id feature is : {} ".format(df_train.shape))
    # print("The test data size before dropping Id feature is : {} ".format(df_test.shape))

    all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
    def dlcall(all_data):
        DLclass = DTPROCESS(all_data)
        DLclass.uselessdata()
        DLclass.categoricaltrchange()
        DLclass.normality()
        DLclass.missdata()
        DLclass.LabelEncoder()
        DLclass.addfeatures()
        DLclass.skewedfeatures()
        DLclass.lookdummies()
        return DLclass.df_train
    all_data = dlcall(all_data)
    train,test = all_data[:ntrain], all_data[ntrain:]
    # print("The train data size after dropping Id feature is : {} ".format(train.shape))
    # print("The test data size after dropping Id feature is : {} ".format(test.shape))
    # print(train.describe())
    # print(y_train) 
    MLClass = ML2(train,test,y_train)
    ensemble = MLClass.startml()
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    print(ensemble)
    sub.to_csv( SCRIPT_PATH + '/submission.csv',index=False)
