# -*- coding: utf-8 -*-
import pandas as pd

train=pd.read_csv('database/train.csv')
test=pd.read_csv('database/test.csv')

#print train.info()
#print test.info()

selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

X_train=train[selected_features]
X_test=test[selected_features]

y_train=train['Survived']

#print X_train['Embarked'].value_counts()
#print X_test['Embarked'].value_counts()

X_train['Embarked'].fillna('S',inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test=dict_vec.fit_transform(X_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
from xgboost import XGBClassifier
xgbc=XGBClassifier()

from sklearn.cross_validation import cross_val_score
#使用5折交叉验证的方法在训练集上默认配置的随机森林进行评估
print cross_val_score(rfc,X_train,y_train,cv=5).mean()
#使用5折交叉验证的方法在训练集上默认配置的随机森林进行评估
print cross_val_score(xgbc,X_train,y_train,cv=5).mean()

rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
rfc_submission.to_csv('database/rfc_submission.csv',index=False)

xgbc.fit(X_train,y_train)
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})
'''
#使用并行网格搜索的方式寻找更好的超参数组合，以期待进一步提高XGBClassifier的预测性能
from sklearn.grid_search import GridSearchCV
params={'max_depth':range(2,7),'n_estimators':range(100,1100,200),'learning_rate':[0.05,0.1,0.25,0.5,1.0]}
xgbc_best=XGBClassifier()
gs=GridSearchCV(xgbc_best,params,n_jobs=-1,cv=5,verbose=1)
gs.fit(X_train,y_train)
xgbc_best_y_predict=gs.predict(X_test)
xgbc_best_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_best_y_predict})
xgbc_best_submission.to_csv('database/xgbc_best_submission.csv',index=False)
'''