# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df = pd.read_csv(path)
df.head(5)
#Code starts here
X = df.drop(columns = "Price")
y = df["Price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 6) 
corr = X_train.corr()
print(corr)



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
pred = regressor.predict(X_test)
r2 = regressor.score(X_test,y_test)
print(round(r2))


# --------------
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# Code starts here
lasso = Lasso(random_state = 0)
lasso.fit(X_train,y_train)
pred = lasso.predict(X_test)
r2_lasso = lasso.score(X_test,y_test)
print(round(r2_lasso,2))



# --------------
from sklearn.linear_model import Ridge

# Code starts here

ridge = Ridge(random_state = 0)
ridge.fit(X_train,y_train)
pred = ridge.predict(X_test)
r2_ridge = ridge.score(X_test,y_test)
print(round(r2_ridge,2))



# --------------
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

#Code starts here
regressor = LinearRegression()
score = cross_val_score(regressor,X_train,y_train,cv =10)
mean_score = score.mean()
print(mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Code starts here
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2_poly  = model.score(X_test,y_test)

print(round(r2_poly,2))


