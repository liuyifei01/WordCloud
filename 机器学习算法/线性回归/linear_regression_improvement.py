import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 加载加利福尼亚的房价数据集
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data["MedHouseVal"] = california.target
# print(data.head())

# 分离特征变量和目标变量
X = data.drop('MedHouseVal', axis=1)            # 删除该列
y = data['MedHouseVal']                               # 只获取该列


# 特征选择
selector = SelectKBest(f_regression, k=6)
X_new = selector.fit_transform(X, y)                # 提取最优的6个特征，并返回6个特征的数据
print(X_new)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_new)
X_test_scaled = scaler.transform(X_test_new)

# 构建新模型
model_new = LinearRegression()
model_new.fit(X_train_scaled, y_train_new)

# 预测
y_train_pred_new = model_new.predict(X_train_scaled)
y_test_pred_new = model_new.predict(X_test_scaled)

# 评估新模型性能
mse_train_new = mean_squared_error(y_train_new, y_train_pred_new)
r2_train_new = r2_score(y_train_new, y_train_pred_new)
mse_test_new = mean_squared_error(y_test_new, y_test_pred_new)
r2_test_new = r2_score(y_test_new, y_test_pred_new)

print(f'Train MSE (new): {mse_train_new}, Train R2 (new): {r2_train_new}')
print(f'Test MSE (new): {mse_test_new}, Test R2 (new): {r2_test_new}')