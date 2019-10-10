# ~*~ coding: utf-8 ~*~
'''
A module with basic techniques for machine learning
with the use of sklearn and other libraries as
xgboost and lightgbm
'''
from pprint import pprint
from datetime import datetime as dt
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
try:
    import lightgbm as lgb
except ImportError as e:
    print(e)


def neg_mse_to_rmse(neg_mse):
    '''
    Converts negative mse to rmse
    '''
    return np.abs(neg_mse) ** 0.5


def f_score_macro(y_true, y_pred):
    ''' Call F1 score with macro averaging '''
    return f1_score(y_true, y_pred, average='macro')
    

def search_grid(model, X_data, y_data, search_space, metric, num_folds, metric_func=None):
    '''
    Searches in the provided space for a best metric
    Returns best model, train/test scores and best parameters
    '''
    grid = GridSearchCV(model, search_space, scoring=metric, cv=num_folds, 
                        n_jobs=-1, return_train_score=True, verbose=2)
    grid.fit(X_data, y_data)
    best_index = grid.best_index_
    best_val_score = grid.best_score_
    best_std = grid.cv_results_['std_test_score'][best_index]
    train_score = grid.cv_results_['mean_train_score'][best_index]
    if metric_func is not None:
        train_score = metric_func(train_score)
        best_val_score = metric_func(best_val_score)
    print('='*5, ' Grid Search Results ', '='*5)
    print('Best parameters are:')
    pprint(grid.best_params_)
    print('Best train score %.4f, validation %.4f (mean val %.4f)' % 
                          (train_score, best_val_score, best_std))
    print('='*33)
    return grid.best_estimator_, grid.best_params_


def plot_feature_importances(feature_names, feature_importances, 
                             orient='h', max_show=15, fig_shape=(8,6)):
    '''
    Plots feature importances
    '''
    feat_imp = pd.DataFrame(
        {'feature': feature_names,
         'importance': feature_importances})
    feat_imp.sort_values('importance', ascending=False, inplace=True)
    plt.figure(figsize=fig_shape)
    plt.title('Feature importances')
    sns.barplot(feat_imp['importance'].iloc[:max_show],
                feat_imp['feature'].iloc[:max_show],
                orient=orient)
    plt.show()

    
def train_model(X_train, y_train, X_val, y_val, estimator, metric_func):
    '''
    Train a model and return it
    '''
    if estimator == 'lgb':
        lgb_train_set = lgb.Dataset(X_train, label=y_train)
        lgb_val_set = lgb.Dataset(X_val, label=y_val)
        estimator = lgb.train(lgbm_params, lgb_train_set, 
                  valid_sets=[lgb_train_set, lgb_val_set], valid_names=['train','valid'],
                  num_boost_round=NUM_ROUNDS, early_stopping_rounds=EARLY_STOP, 
                  verbose_eval=NUM_ROUNDS//10)
    else:
        estimator.fit(X_train, y_train)
    train_predicted,val_predicted = estimator.predict(X_train), estimator.predict(X_val)
    train_score = metric_func(y_train, train_predicted)
    val_score = metric_func(y_val, val_predicted)
    return estimator, train_score, val_score


def balanced_cross_validation(X_data, y_data, estimator, num_folds, metric_func,
                              train_factor=1, val_factor=1, minor_class=1, name='model'):
    '''
    Performs cross-validation with balancing data
    ------
    Pseudocode:
    for each split:
        select train/valid positive labels and their features at split indices
        select train/valid negative labels and their features at random with given size
        concatenate positive and negative into train/valid sets
        train model and save train/valid scores
    Return cross-validated models and scores
    '''
    pos_data,pos_labels = X_data[y_data == minor_class], y_data[y_data == minor_class]
    neg_data,neg_labels = X_data[y_data != minor_class], y_data[y_data != minor_class]
    kfold = KFold(n_splits=num_folds, shuffle=True)
    cv_models, cv_scores = {}, {'train': {}, 'val': {}}
    for i, (train_i,val_i) in enumerate(kfold.split(pos_labels)):
        train_pos,val_pos = pos_data.iloc[train_i],pos_data.iloc[val_i]
        train_pos_lab,val_pos_lab = pos_labels.iloc[train_i],pos_labels.iloc[val_i]
        train_size,val_size = int(len(train_i)*train_factor), int(len(val_i)*val_factor)
        train_neg_i = np.random.choice(range(len(neg_labels)), size=train_size, replace=False)
        val_neg_i = np.random.choice(range(len(neg_labels)), size=val_size, replace=False)
        train_neg,val_neg = neg_data.iloc[train_neg_i],neg_data.iloc[val_neg_i]
        train_neg_lab,val_neg_lab = neg_labels.iloc[train_neg_i],neg_labels.iloc[val_neg_i]
        X_train = pd.concat([train_pos, train_neg], axis=0, ignore_index=True)
        y_train = pd.concat([train_pos_lab, train_neg_lab], axis=0, ignore_index=True,)
        X_val = pd.concat([val_pos, val_neg], axis=0, ignore_index=True)
        y_val = pd.concat([val_pos_lab, val_neg_lab], axis=0, ignore_index=True,)
        model,train_score,val_score = train_model(X_train, y_train, X_val, y_val, 
                                                  estimator, metric_func)
        new_name = name + str(i)
        cv_models[new_name] = model
        cv_scores['train'][new_name],cv_scores['val'][new_name] = train_score, val_score
    return cv_models, cv_scores


def isolation_forest_f1(y_true, y_predicted):
    '''
    Custom calculation of F1 score for isolation forest
    '''
    isf_pred = np.where(y_predicted == -1, 1, 0)
    return f1_score(y_true, isf_pred)


def lgb_f1(y_true, y_predicted):
    '''
    Computes F1 for lgb probability predictions
    '''
    threshold = 0.5
    y_predicted = np.where(y_predicted > threshold, 1, 0)
    return f1_score(y_true, y_predicted)


def cross_validation(train_data, train_labels, estimator, num_folds, metric_func,
                     model_name='model', stratified=False, **kwargs
                     ):
    '''
    Performs num_folds cross-validation using estimator
    Returns a dictionary of trained models and scores
    (If using lightgbm then provide 'lgb' to estimator and 
    make sure you imported it as "import lightgbm as lgb",
    also provide lgbm_params, num_rounds, early_stop;
    and metric_fn should be str the same as in lgbm_params)
    '''
    if not stratified:
        cv = KFold(n_splits=num_folds, shuffle=True, random_state=100)
    else:
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=100)
    cv_models = {}
    scores = {'train': {},
              'val': {}}

    for i,(train_i,val_i) in tqdm(enumerate(cv.split(train_data, train_labels)), desc='Folds'):
        if isinstance(train_data, (pd.core.frame.DataFrame, pd.core.series.Series)):
            X_train,y_train = train_data.iloc[train_i], train_labels[train_i]
            X_val,y_val = train_data.iloc[val_i], train_labels[val_i]
        elif isinstance(train_data, (np.ndarray,)):
            X_train,y_train = train_data[train_i], train_labels[train_i]
            X_val,y_val = train_data[val_i], train_labels[val_i]
        else:
            raise Exception('Passed data not pd.DataFrame, pd.Series or np.ndarray')
        if estimator == 'lgb':
            assert 'lgbm_params' in kwargs.keys(), 'If estimator lgb then provide lgbm_params in kwargs'
            assert 'num_rounds' in kwargs.keys(), 'If estimator lgb then provide num_rounds in kwargs'
            assert 'early_stop' in kwargs.keys(), 'If estimator lgb then provide early_stop in kwargs'
            lgb_train_set = lgb.Dataset(X_train, label=y_train)
            lgb_val_set = lgb.Dataset(X_val, label=y_val)
            booster = lgb.train(kwargs['lgbm_params'], lgb_train_set, 
                      valid_sets=[lgb_train_set, lgb_val_set], valid_names=['train','valid'],
                      num_boost_round=kwargs['num_rounds'], early_stopping_rounds=kwargs['early_stop'], 
                      verbose_eval=kwargs['num_rounds']//10)
            new_name = model_name + str(i)
            cv_models[new_name] = booster
            train_predicted = booster.predict(X_train)
            val_predicted = booster.predict(X_val)
            scores['train'][new_name] = metric_func(y_train, train_predicted)
            scores['val'][new_name] = metric_func(y_val, val_predicted)
#             scores['train'][new_name] = booster.best_score['train'][metric_func] #** 0.5
#             scores['val'][new_name] = booster.best_score['valid'][metric_func] #** 0.5
        else:
            estimator.fit(X_train, y_train)
            new_name = model_name + str(i)
            cv_models[new_name] = estimator
            train_predicted = estimator.predict(X_train)
            val_predicted = estimator.predict(X_val)
            scores['train'][new_name] = metric_func(y_train, train_predicted)
            scores['val'][new_name] = metric_func(y_val, val_predicted)
    return cv_models, scores


def cv_predict(cv_models, x_data):
    '''
    Makes average prediction cross-validated models
    '''
    cv_predicted = np.zeros((len(x_data),))
    for _,mod in cv_models.items():
        cv_predicted += mod.predict(x_data,) / len(cv_models)
    return cv_predicted
	
	
def holdout_indices(start_range, stop_range, range_step, holdout_len, print_stats=True):
    '''
    Init empty list
    For index in (start_range, stop_range, range_step)
        select random start index within (start_range, stop_range - holdout_len)
        assing end index by adding houldout_len
        update the list with selected index range
    Returns an array of selected indices
    '''
    ar = []
    for i in range(start_range, stop_range, range_step):
        end_index = stop_range + 1
        rep = 0
        while end_index > stop_range:
            start_index = np.random.randint(i, i + range_step - holdout_len)
            end_index = start_index + holdout_len
            ar.extend(list(np.arange(start_index, end_index)))
            rep += 1
            if rep > 5:
                error_msg = ('Unable to select end_index lower than stop_range. '
							 'Provide different range_step and holdout_len')
                raise Exception(error_msg)
    if print_stats:
        print('{} indices selected'.format(len(ar)))
    return np.array(ar)


def quick_ml(models, train_data, val_data, metric_fn):
    '''
    Quickly test different ML algorithms and compare scores on train and valid
    ------
    Params:
        models: a dictionary of model names and their callables
        features: a dataframe of features
        labels: an array of labels
        val_size: a portion of data to allocate to validation
    '''
    X_train,y_train = train_data
    X_val,y_val = val_data
    scores = {}
    trained = {}
    for m_name,model in models.items():
        model.fit(X_train,y_train)
        train_score = metric_fn(y_train, model.predict(X_train))
        val_score = metric_fn(y_val, model.predict(X_val))
        scores[m_name] = (train_score, val_score)
        trained[m_name] = copy.deepcopy(model)
    scores = pd.DataFrame(scores, index=['train','val']).T
    return trained,scores


def reduce_dimensions(*data, decomposer, **kwargs):
    '''
    Reduce dimensions of data for training or visualization
    Use first data argument as the one for fitting, others are for transformation only
    -----
    Return fitted algorithm and decomposed data as a list
    '''
    decomposer = decomposer.lower()
    assert decomposer in ['pca','svd','tsne']
    if decomposer == 'pca':
        print('Fitting PCA with', kwargs)
        decomposer = PCA(**kwargs)
    elif decomposer == 'svd':
        print('Fitting Truncated SVD with', kwargs)
        decomposer = TruncatedSVD(**kwargs)
    elif decomposer == 'tsne':
        print('Fitting T-SNE with', kwargs)
        decomposer = TSNE(**kwargs)
    decomposer.fit(data[0])
    transformed = []
    for d in data:
        t = decomposer.transform(d)
        transformed.append(t)
    print(f'Explained variance ratio {decomposer.explained_variance_ratio_.sum():.3f}')
    return decomposer, transformed


def scale_features(*data, scaler, **kwargs):
    '''
    Rescale data
    Use first data argument as the one for fitting, others are for transformation only
    -----
    Return fitted algorithm and decomposed data as a list
    '''
    scaler = scaler.lower()
    assert scaler in ['minmax','standard']
    if scaler == 'minmax':
        print('Min-Max scaling with', kwargs)
        scaler = MinMaxScaler(**kwargs)
    elif scaler == 'standard':
        print('Standard scaling with', kwargs)
        scaler = StandardScaler(**kwargs)
    scaler.fit(data[0])
    transformed = []
    for d in data:
        t = scaler.transform(d)
        transformed.append(t)
        print(f'New mean and std are {t.mean():.4f} +- {t.std():.5f}')
    return scaler, transformed