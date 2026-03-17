import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from preprocessor import preprocessor


df = pd.read_csv('data_cleaned.csv')

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in split.split(df, df['income']):
    strat_train_set = df.loc[train_idx]
    strat_test_set = df.loc[test_idx]

X_train = strat_train_set.drop("median_house_value", axis=1)
y_train = strat_train_set["median_house_value"]

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"]

y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)


y_train = np.log1p(y_train_original)
y_test = np.log1p(y_test_original)

X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

xgb_model = XGBRegressor(
    subsample= 1.0,
    reg_lambda= 1,
    reg_alpha= 0.1,
    n_estimators= 800,
    min_child_weight= 1,
    max_depth= 7,
    learning_rate= 0.05,
    gamma= 0,
    colsample_bytree= 1.0
)

xgb_model.fit(X_train_prepared, y_train)

y_pred_log = xgb_model.predict(X_test_prepared)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_original, y_pred)
print("MAE:", mae)

joblib.dump(preprocessor, "preprocessor.joblib")
joblib.dump(xgb_model, "xgb_model.joblib")

print("Saved preprocessor.joblib and xgb_model.joblib")
