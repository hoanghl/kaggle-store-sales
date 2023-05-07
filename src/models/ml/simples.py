# import xgboost as xgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_regression(Xtrain, ytrain, Xtest, ytest, **kwargs):
    regression = LinearRegression(positive=True, n_jobs=10)
    regression.fit(Xtrain, ytrain)

    pred = regression.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")


def random_forest(Xtrain, ytrain, Xtest, ytest, **kwargs):
    regression = RandomForestRegressor(n_estimators=100, random_state=37, n_jobs=10)
    regression.fit(Xtrain, ytrain)

    pred = regression.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")


def xgboost(Xtrain, ytrain, Xtest, ytest, **kwargs):
    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
    reg.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], verbose=False)

    pred = reg.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")
