from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV
import pandas as pd
from sklearn.metrics import mean_squared_error ,r2_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np



datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dummies=pd.get_dummies(datas[["NewLeague","League","Division"]])
y=datas["Salary"]
x_=datas.drop(["NewLeague","League","Division","Salary"],axis=1).astype("float64")
x=pd.concat([x_,dummies],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

lambdalar=10**(np.linspace(10,-2,100))
ElasticNetReg=ElasticNet()
ElastiCV=ElasticNetCV(alphas=lambdalar,cv=10,max_iter=10000).fit(x_train,y_train)
opt_lambda=ElastiCV.alpha_
ElasticNetRegTuned=ElasticNet(alpha=opt_lambda).fit(x_train,y_train)
predict=ElasticNetRegTuned.predict(x_test)
RMSE=np.sqrt(np.mean(mean_squared_error(y_test,predict)))
R2=r2_score(y_test,predict)
print(RMSE)






















