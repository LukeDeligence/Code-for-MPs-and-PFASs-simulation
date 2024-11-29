import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import shap
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
# Import Data, please import the data form the flods saving the data
#microplastics = pd.read_excel("combined Data.xlsx", sheet_name="MPs")
#PFASs = pd.read_excel("combined Data.xlsx", sheet_name="PFASs")
#env =  pd.read_excel("combined Data.xlsx", sheet_name="env")
#depth_labels = pd.read_excel("combined Data.xlsx", sheet_name="classfication")
X = pd.concat([PFASs,env],axis=1)#This is for MPs model construct
#X = pd.concat([microplastics,env],axis=1)#This is for PFASs model construct
X = pd.DataFrame(X)
y = microplastics.iloc[:,0]#This is for MPs model construct
#y = PFASs.iloc[:,5]#This is for PFASs model construct
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=depth_labels, train_size=0.8, test_size=0.2, random_state=42)
# Random forest
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt']
}
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5,
                                   scoring='neg_mean_squared_error', verbose=2, n_jobs=1, random_state=42)
random_search.fit(X_train, y_train)
# 输出最佳参数
print("Best parameters for Random Forest:", random_search.best_params_)
rf_best = RandomForestRegressor(**random_search.best_params_, random_state=42)
rf_best.fit(X_train, y_train)
y_pred_rf = rf_best.predict(X_test)

# SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}
svm = SVR()
grid_search_svm =RandomizedSearchCV(estimator=svm,param_distributions=param_grid_svm, n_iter=100,cv=5,
                               scoring='neg_mean_squared_error', verbose=2, n_jobs=1)
grid_search_svm.fit(X_train, y_train)
print("Best parameters for SVM:", grid_search_svm.best_params_)
svm_best = SVR(**grid_search_svm.best_params_)
svm_best.fit(X_train, y_train)
y_pred_svm = svm_best.predict(X_test)
#XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_model = xgb.XGBRegressor(random_state=42)
grid_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid_xgb, n_iter=100,cv=5,
                               scoring='neg_mean_squared_error', verbose=2, n_jobs=1)
grid_search_xgb.fit(X_train, y_train)
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
xgb_best = xgb.XGBRegressor(**grid_search_xgb.best_params_, random_state=42)
xgb_best.fit(X_train, y_train)
y_pred_xgb = xgb_best.predict(X_test)
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
print("SVM MSE:", mean_squared_error(y_test, y_pred_svm))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

# Based on the model results, select the best model to make IML
# Random forest result
y_train_pred_rf = rf_best.predict(X_train)
# SVM result
y_train_pred_svm = svm_best.predict(X_train)
# XGBoost result
y_train_pred_xgb = xgb_best.predict(X_train)
#
traintest = pd.DataFrame({
})
TrainResults = pd.DataFrame({
    'RF train Predictions': y_train_pred_rf,
    'SVM train Predictions': y_train_pred_svm,
    'XGBoost train Predictions': y_train_pred_xgb,
    #'Training Data X': X_train,
    'Training Data Y': y_train,
})
TrainResults.to_excel('Train.xlsx', index=False)
TestResults = pd.DataFrame({
    'RF test Predictions': y_pred_rf,
    'SVM test Predictions': y_pred_svm,
    'XGBoost test Predictions': y_pred_xgb,
    #'Test Data X': X_test,
    'Test Data Y': y_test
})
TestResults.to_excel('Test.xlsx', index=False)



best_rf = grid_search_xgb.best_estimator_
predictions = best_rf.predict(X_test)
best_mse = mean_squared_error(y_test, predictions)
best_r2 = r2_score(y_test, predictions)
print('MSE of the best model: ', best_mse)
print('R^2 of the best model: ', best_r2)

# 可视化特征重要性
feature_importances = best_rf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[-10:]
important_features = feature_importances[sorted_idx]
important_feature_names = feature_names[sorted_idx]

# 对筛选后的特征重要性进行排序
sorted_important_idx = np.argsort(important_features)

# visialization
plt.figure(figsize=(4, 8))
plt.barh(range(len(sorted_important_idx)), important_features[sorted_important_idx], color='skyblue')
plt.yticks(range(len(sorted_important_idx)), important_feature_names[sorted_important_idx], fontsize=12)
plt.xlabel("Feature Importance", fontsize=16)
for i, v in enumerate(important_features[sorted_important_idx]):
    plt.text(v + 0.001, i, f"{v:.3f}", fontsize=12)
# 计算并显示图中变量合计的重要性
total_importance = np.sum(important_features)
plt.figtext(0.15, 0.15, f"Total Importance of Displayed Features: {total_importance:.3f}", ha="center", fontsize=12)
#plt.savefig('top_10_feature_importance.pdf', bbox_inches='tight')
plt.close()
print(feature_importances)

sorted_idx = np.argsort(feature_importances)[-2:]
important_features = feature_importances[sorted_idx]
important_feature_names = feature_names[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 6))
PartialDependenceDisplay.from_estimator(
    best_rf,
    X,
    features=[sorted_idx],
    feature_names=feature_names,
    grid_resolution=50,
    ax=ax
)
plt.savefig('PDP_.pdf', format='pdf', bbox_inches='tight')
plt.close()
#SHAP
explainer = shap.Explainer(best_rf, X)
shap_values = explainer(X)
plt.figure(figsize=(4, 8))
shap.summary_plot(shap_values, X, max_display=10,show=False)
plt.savefig('SHAP_.pdf', format='pdf', bbox_inches='tight')
plt.close()

#PDP
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for i, idx in enumerate(sorted_idx):
    PartialDependenceDisplay.from_estimator(
        best_rf,
        X,
        features=[idx],
        feature_names=feature_names,
        grid_resolution=50,
        ax=ax[i]
    )
    ax[i].set_title(f"PDP for {feature_names[idx]}")
plt.suptitle("Partial Dependence Plots for Top 2 Important Features", fontsize=16)
plt.subplots_adjust(top=0.85)  # 调整标题位置
# 保存为PDF格式
plt.savefig('pdp_top_2_features.pdf', format='pdf', bbox_inches='tight')
plt.close()


print('Finished!')