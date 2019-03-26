from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from catboost import CatBoostRegressor


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def load_X_train():
    print('Load X_train')
    X_train = pd.read_csv('nlmk_public/X_train.csv', sep=',')
    X_train = X_train[1:].astype('float')
    return X_train


def load_y_train():
    print('Load y_train')
    y_train = pd.read_csv('nlmk_public/y_train.csv', sep=',')
    return y_train


def load_X_test():
    print('Load X_train')
    X_test = pd.read_csv('nlmk_public/X_test.csv', sep=',')
    X_test = X_test.astype('float')
    return X_test


def filling_na_train():
    print('Counting N/A for X_train')
    X_train = load_X_train()
    X_train_rows = len(X_train.index)
    X_train_count = X_train.count()
    X_train_na = X_train_count[X_train_count < X_train_rows]
    print(X_train_na.sort_values().apply(lambda x: (X_train_rows - x) / X_train_rows))
    print('Filling N/A')
    X_train = X_train.fillna(0)
    return X_train


def filling_na_test():
    print('Counting N/A for X_test')
    X_test = load_X_test()
    X_test_rows = len(X_test.index)
    X_test_count = X_test.count()
    X_test_na = X_test_count[X_test_count < X_test_rows]
    print(X_test_na.sort_values().apply(lambda x: (X_test_rows - x) / X_test_rows))
    print('Filling N/A')
    X_test = X_test.fillna(0)
    return X_test


def filling_na_mean_train():
    print('Counting N/A for X_train')
    X_train = load_X_train()
    X_train_rows = len(X_train.index)
    X_train_count = X_train.count()
    X_train_na = X_train_count[X_train_count < X_train_rows]
    print(X_train_na.sort_values().apply(lambda x: (X_train_rows - x) / X_train_rows))
    print('Filling N/A by mean')
    X_train = X_train.fillna(X_train.mean())
    return X_train


def filling_na_mean_test():
    print('Counting N/A for X_test')
    X_test = load_X_test()
    X_test_rows = len(X_test.index)
    X_test_count = X_test.count()
    X_test_na = X_test_count[X_test_count < X_test_rows]
    print(X_test_na.sort_values().apply(lambda x: (X_test_rows - x) / X_test_rows))
    print('Filling N/A by mean')
    X_test = X_test.fillna(X_test.mean())
    return X_test


def feature_engineering(some_df):
    print('Feature Engineering')
    print('NANs: ', some_df.isnull().sum().sum())
    ll = ['speed', 'force', 'temperature', 'power', 'chem', 'gap', 'size']
    cols = set(some_df.columns)
    for l in ll:
        for i in range(0, 100):
            c1 = str(l) + str(i)
            c2 = str(l) + str(i-1)
            if c1 not in cols:
                continue
            if c2 not in cols:
                continue
            some_df[c1 + '_' + c2] = some_df[c2] / some_df[c1]
            some_df[c1 + '_plus_' + c2] = some_df[c1] + some_df[c2]
            some_df[c1 + '_minus_' + c2] = some_df[c1] - some_df[c2]
    #some_df = some_df.fillna(0)
    print('NANs: ', some_df.isnull().sum().sum())
    print('Feature Engineering Ended')
    return some_df


def drop_features(X_train, X_test):
    columns = ['chem14', 'chem32', 'chem28', 'chem33', 'chem25', 'chem22', 'chem20', 'chem11', 'chem13', 'chem12',
               'chem8', 'chem7', 'chem4', 'force26', 'chem31']
    X_train = X_train.drop(columns, axis=1)
    X_test = X_test.drop(columns, axis=1)
    return X_train, X_test


def normalization(X_train, X_test):
    print('Normalization')
    X_train = X_train
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_train)
    return X_train


def CV_KFold(X_train, y_train):
    print('k-fold Cross-Validation')
    # k-fold cross validation evaluation of xgboost model
    # CV model
    xgb = XGBRegressor(silent=False, nthread=10, max_depth=8, n_estimators=200, subsample=0.5, learning_rate=0.1,
                       seed=200, random_state=42)
    kfold = KFold(n_splits=5, random_state=42)
    results = cross_val_score(xgb, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    #preds = cross_val_predict(xgb, X_train, y_train, cv=kfold)
    #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    print('MSE:', results.mean())


def xgb_feature_importance(model):
    return model.feature_importances_


def plot_feature_importance(model, X_train):
    # Plot feature importance
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def xgb_RSearchCV(X_train, y_train, X_test):
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [2, 4, 6, 8]
    }
    xgb = XGBRegressor(learning_rate=0.02, n_estimators=600, silent=True, nthread=1,
                       seed=200, random_state=42)
    folds = 4
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='neg_mean_squared_error', n_jobs=4,
                                       cv=skf.split(X=X_train, y=y_train), verbose=3, random_state=42)
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    random_search.fit(X=X_train, y=y_train)
    timer(start_time)  # timing ends here for "start_time" variable
    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    preds = random_search.predict(X_test)
    return preds


def gbm_regressor(X_train, y_train, X_test):
    print('Gradient Boosting Model (sklearn)')
    params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.1, 'loss': 'ls', 'random_state': 42}
    model = GradientBoostingRegressor(**params)
    model.fit(X=X_train, y=y_train)
    #plot_feature_importance(model=model, X_train=X_train)
    preds = model.predict(X_test)
    #mse_br = mean_squared_error(y_test, preds)
    #print("Graidient Boost MSE: %.4f" % mse_br)
    return preds


def xgb_regressor(X_train, y_train, X_test):
    print('XGBoost Regressor')
    params = {'nthread': 10, 'max_depth': 9, 'n_estimators': 3000, 'subsample': 0.9, 'gamma': 0.1,
              'learning_rate': 0.01, 'seed': 0, 'random_state': 42, 'predictor': 'cpu_predictor', 'eval_metric': 'rmse'}
    xgb = XGBRegressor(silent=False, **params)
    xgb.fit(X=X_train, y=y_train)
    preds = xgb.predict(X_test)
    return preds


def catboost_regressor(X_train, y_train, X_test):
    print('Catboost Regressor')
    model = CatBoostRegressor(iterations=250000, random_state=42, loss_function='MAE')
    model.fit(X=X_train, y=y_train, verbose=200)
    preds = model.predict(X_test)
    return preds


def submit_predictions(preds):
    with open('answers.txt', 'w') as output:
        for i in range(preds.shape[0]):
            output.write('%f\n'%preds[i])
    print('Submit Ok')


def submit_predictions2(preds):
    preds.to_csv('stacking_xgb_cbt.csv', header=None, index=None)
    print('Submit Ok')


def submit(df):
    df.to_csv('df.csv', index=None)
    print('Submit Ok')


def stacking():
    submit_xgboost = pd.read_csv('nlmk_public/answers_xgb_3000.csv', header=None)
    submit_catboost = pd.read_csv('nlmk_public/answers_cbt_125000.csv', header=None)
    submit = pd.concat([submit_xgboost, submit_catboost], axis=1)
    submit = submit.mean(axis=1)
    return submit


def main():
    print('RAIF: https://raif.jet.su/profile/')
    start_time = timer(None)
    y_train = load_y_train()
    #X_train = filling_na_train()
    #X_test = filling_na_test()
    X_train = filling_na_mean_train()
    X_test = filling_na_mean_test()
    X_train, X_test = drop_features(X_train=X_train, X_test=X_test)
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)
    #submit(X_train)

    #CV_KFold(X_train=X_train, y_train=y_train)
    #preds = gbm_regressor(X_train=X_train, y_train=y_train, X_test=X_test)
    #preds = xgb_regressor(X_train=X_train, y_train=y_train, X_test=X_test)
    #preds = xgb_GridSearchCV(X_train=X_train, y_train=y_train, X_test=X_test)
    preds = catboost_regressor(X_train=X_train, y_train=y_train, X_test=X_test)

    submit_predictions(preds)
    timer(start_time)
    print('Ok')


if __name__ == '__main__':
    main()
