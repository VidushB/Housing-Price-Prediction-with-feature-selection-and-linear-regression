from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
model = LinearRegression()
X=df_train.drop("SalePrice",axis=1)
y = df_train['SalePrice'].reset_index(drop = True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train,y_train)
r_sq=model.score(X_train,y_train)
print("Co-efficient of determination is : ",r_sq)

y_predict=model.predict(X_val)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_predict)) 
