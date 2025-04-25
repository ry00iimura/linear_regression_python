import sys #1つ上のフォルダ内のモジュールをインポート
sys.path.append('../') #1つ上のフォルダ内のモジュールをインポート
from data_preprocessing_python.generic_df_process import ML 
import seaborn as sns
from sklearn import linear_model
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVC,NuSVC,LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

class LinearRegression(ML):
    'linear regressions'

    def mlr(self):
        'multi linear regression'
        self.clf = linear_model.LinearRegression()
        # 説明変数
        X = self.train_X.values
        # 目的変数
        Y = self.train_y.values
        # 予測モデルを作成
        self.clf.fit(X, Y)
        # 予測モデルを作成
        self.clf.score(X, Y)
        # 偏回帰係数
        coff= (pd.DataFrame({'Name':self.train_X.columns,
                            "Coefficients":np.abs(self.clf.coef_)}
                            ).sort_values(by='Coefficients')
                            )

        # 切片 (誤差)
        red = (self.clf.intercept_)
        
        return coff,red

    def glm(self,regressor):
        '''
        general linear models
        regressor example: https://www.statsmodels.org/stable/glm.html
        sm.families.Binomial() #logstic regression
        sm.families.Gaussian() #Gaussian regression
        sm.families.Poisson() # Poisson distribution
        '''
        self.clf = sm.GLM(self.train_y, self.train_X, family=regressor)
        glm_result = self.clf.fit()
        print(glm_result.summary())

    def lasso(self):
        'lasso'
        self.clf= linear_model.Lasso(alpha=1.0)
        self.clf.fit(self.train_X,self.train_y)
        print("\nLassoでの係数")
        print(self.clf.intercept_) 
        print(self.clf.coef_)

    def ridge(self):
        'ridge'
        self.clf= linear_model.Ridge(alpha=1.0)
        self.clf.fit(self.train_X,self.train_y)
        print("\nRidgeでの係数")
        print(self.clf.intercept_) 
        print(self.clf.coef_)

    def elastic(self):
        'elasticNet'
        self.clf= linear_model.ElasticNet(alpha=1.0)
        self.clf.fit(self.train_X,self.train_y)
        print("\nElasticNetでの係数")
        print(self.clf.intercept_) 
        print(self.clf.coef_)

    def multiColinearity(self):
        'vizualize the correlation strength between two independent variables'
        plt.figure(figsize=(30,30))
        sns.heatmap(self.train_X.corr(),annot_kws={'size':15},annot=True)

class Logistic(ML):
    'LogisticRegressionCV'

    def GS(self,cs,solver,maxIter,classWeight):
        '''
        grid search on LogisticRegressionCV
        param e.g.)
        {'Cs':[10,50],'solver':['newton-cg','lbfgs','liblinear','sag','saga'],'max_iter':[1000],'class_weight':['balanced']}
        '''
        self.clf = linear_model.LogisticRegressionCV()
        params_dict = {'Cs':cs,'solver':solver,'max_iter':maxIter,'class_weight':classWeight,'random_state':[5]}
        self.grid_search = GridSearchCV(self.clf,  # 分類器を渡す
                                param_grid=params_dict,  # 試行してほしいパラメータを渡す
                                cv=10,  # 10-Fold CV で汎化性能を調べる
                                )
        self.grid_search.fit(self.train_X, self.train_y)
        print(self.grid_search.best_score_)  # 最も良かったスコア
        print(self.grid_search.best_params_)  # 上記を記録したパラメータの組み合わせ
        return self.grid_search.best_params_.values()

class SVM(ML):
    'suport vector machine'

    def GS(self,loss,classWeight,maxIter):
        '''
        grid search on LinearSVC
        param e.g.)
        {'loss':['hine','squared_hinge'],'class_weight':['balanced'],'max_iter':[5000]}
        '''
        self.clf = LinearSVC()
        params_dict = {'loss':loss,'class_weight':classWeight,'random_state':[1],'max_iter':maxIter}
        self.grid_search = GridSearchCV(self.clf,  # 分類器を渡す
                                param_grid=params_dict,  # 試行してほしいパラメータを渡す
                                cv=10,  # 10-Fold CV で汎化性能を調べる
                                )
        self.grid_search.fit(self.train_X, self.train_y)
        print(self.grid_search.best_score_)  # 最も良かったスコア
        print(self.grid_search.best_params_)  # 上記を記録したパラメータの組み合わせ
        return self.grid_search.best_params_.values()

# for debugging
# import pydataset
# dataset = pydataset.data('cbpp')
# independent = ['herd','incidence','size']
# dependent = 'period'
