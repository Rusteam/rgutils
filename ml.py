	
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from pprint import pprint
from datetime import datetime as dt
import pandas as pd
import numpy as np
import copy


def get_date_range(ym_series):
    '''
    Get a series of YYYY-mm data and find out a max date
    '''
    date_series = ym_series.apply(lambda x: x + '-01')
    date_series = date_series.apply(lambda x: dt.strptime(x, '%Y-%m-%d'))
    return date_series.agg(['min','max'])


def feature_distribution(feature_name, tables, dist_type='value_counts', sort_index=True):
    '''
    Compares distribution of a feature across train and test
    Distribution types are value_counts, describe
    '''
    assert isinstance(tables, (list, tuple)) 
    tbl_dists = []
    if dist_type == 'value_counts':
        for tbl in tables:
            tbl_dist = tbl[feature_name].value_counts(normalize=True)
            tbl_dists.append(tbl_dist)
    elif dist_type == 'describe':
        for tbl in tables:
            tbl_dist = tbl[feature_name].describe()
            tbl_dists.append(tbl_dist)
    dist = pd.concat(tbl_dists, axis=1, join='outer', sort=sort_index)
    dist.columns = ['table_{}'.format(i) for i in range(len(dist.columns))]
    dist.index.name = feature_name
    return dist


def reduce_memory_usage(data, col_types):
    '''
    Converts cols to more memory efficient types
    '''
    print("Memory consumption before conversion {} mb".format(round(data.memory_usage().sum()/1024**2, 1)))
    for c,t in col_types.items():
        data[c] = data[c].astype(t)
    print("Memory consumption after conversion {} mb".format(round(data.memory_usage().sum()/1024**2, 1)))
    return data


def impute_missing(data, impute_values, print_summary=True):
    '''
    Impute set values into missing
    '''
    for col,imp_val in impute_values.items():
        num_missing = data[col].isna().sum()
        data[col].fillna(imp_val, inplace=True)
        if print_summary:
            print('{} missing values were replaced in {}'.format(num_missing, col))
    return data


def ohe_encoding(feature_name, data, suffix='ohe'):
    '''
    One hot encoding for a feature
    '''
    ohe = pd.get_dummies(data[feature_name])
    ohe.columns = ['_'.join([feature_name,suffix,str(col)]) for col in ohe.columns]
    return ohe


def mean_encoding(feature_name, data, target_name, func='mean', 
                  is_test=False, mean_encodings=None, suffix='mean_enc'):
    '''
    Performs mean encoding for a categorical feature
    If train then compute means and fill instead of categories
    Else if test then use precomputed mean encodings to fill
    '''
    if not is_test:
        mean_encodings = data.groupby(feature_name,)\
                                .agg({target_name:func})\
                                .to_dict()[target_name]
    else:
        assert isinstance(mean_encodings, dict)
    encoded = data[feature_name].map(mean_encodings)
    encoded.name = '_'.join([feature_name, suffix])
    return encoded, mean_encodings


def extract_period(colname, data, year=True, month=True, day=True, quarter=False,
                   input_format='%Y-%m-%d'):
    '''
    Extracts periods such as year, month and day of month from a date series
    '''
    datetime_series = data[colname].apply(lambda x: dt.strptime(x, input_format))
    include_periods = {}
    if year:
        y_series = datetime_series.apply(lambda x: x.year)
        include_periods['year'] = y_series
    if month:
        m_series = datetime_series.apply(lambda x: x.month)
        include_periods['month'] = m_series
    if day:
        d_series = datetime_series.apply(lambda x: x.day)
        include_periods['day'] = d_series
    if quarter:
        m_series = datetime_series.apply(lambda x: x.month)
        q_series = m_series.apply(lambda x: x // 3 + int(x % 3 != 0))
        include_periods['quarter'] = q_series
    return pd.DataFrame(include_periods)


def time_diff(colname, data, date_end, diff_base='day', input_format='%Y-%m-%d', 
              suffix='diff'):
    '''
    Computes time difference between date series and date_end 
    with specified base: year/month/day/hour/minute/second
    '''
    assert diff_base in ['year','month','day','hour','minute','second']
    denominators = {'year': 365.25, 'month': 30.4375, 'day': 1,
                    'hour': 60**2, 'minute': 60, 'second': 1}
    dt_start = data[colname].apply(lambda x: dt.strptime(x, input_format))
    dt_end = dt.strptime(date_end, input_format)
    if diff_base in ['year','month','day']:
        dt_diff = dt_start.apply(lambda x: (dt_end - x).days // denominators[diff_base])
    else:
        dt_diff = dt_start.apply(lambda x: (dt_end - x).total_seconds() // denominators[diff_base])
    dt_diff.name = '_'.join([diff_base, suffix])
    return dt_diff


def clip_features(data, col_min_max):
    '''
    Clips each feature setting new min and max
    '''
    for c,(mini,maxi) in col_min_max.items():
        data[c+'_clipped'] = np.clip(data[c], mini, maxi)
    return data


def log_features(data, cols_to_log):
    '''
    Log1p transforms each provided feature
    '''
    for c in cols_to_log:
        data[c+'_log'] = np.log1p(data[c],)
    return data


def mark_anomaly(data, features_normal, agg_funcs=['sum'], suffix='outlier', print_stats=True):
    '''
    Marks all values as (ab)normal in every feature
    '''
    q_upper = '{} <= {}'
    q_lower_upper = '{} <= {} <= {}'
    for feat, bounds in features_normal.items():
        new_name = '_'.join([feat, suffix])
        if isinstance(bounds, (list,tuple)):
            data[new_name] = ((data[feat] < bounds[0]) | (data[feat] > bounds[1])).astype(np.int8)
            if print_stats:
                print('Percent of outliers in {} is {}'.format(feat, data[new_name].mean()))
        elif isinstance(bounds, (int, float)):
            data[new_name] = (data[feat] > bounds).astype(np.int8)
            if print_stats:
                print('Percent of outliers in {} is {}'.format(feat, data[new_name].mean()))
        else:
            raise Exception('Provide correct boundary for {}'.format(feat))
    outlier_features = ['_'.join([feat,suffix]) for feat in features_normal.keys()]
    for agg in agg_funcs:
        data['_'.join([suffix,agg])] = data[outlier_features].agg(agg,axis=1)
    return data


def pairwised_distributions(data, index_column, value_column):
    '''
    Create a dict with unique keys from index column
    and its subset of value column in data table
    '''
    indices = data[index_column].unique()
    data_dict = {}
    for ind in indices:
        subset = data[data[index_column] == ind]
        data_dict[ind] = subset[value_column]
    return data_dict


def calc_shares(numeric_feature, category_feature, id_colname, data, 
               percentage=True, absolute=True, suffix='', eps=1e-4):
    '''
    Computes percentages and absolute values of a numeric feature 
    across different levels of a categorical feature
    for each value in ID colname
    '''
    
    pt = data.pivot_table(index=id_colname, columns=category_feature,
                          values=numeric_feature, aggfunc=np.sum, fill_value=0)
    cat_names = data[category_feature].unique().tolist()
    if percentage:
        pt['total'] = pt[cat_names].sum(axis=1)
        for cat in cat_names:
            new_name = '_'.join([category_feature,str(cat),'percent',suffix]).rstrip('_')
            pt[new_name] = pt[cat] / (pt['total'] + eps)
        del pt['total']
    new_cat_names = []
    if not absolute:
        for cat in cat_names:
            del pt[cat]
    else:
        for cat in cat_names:
            new_name = '_'.join([category_feature, str(cat), suffix]).rstrip('_')
            new_cat_names.append(new_name)
            pt.rename({cat: new_name}, axis='columns', inplace=True)
    pt.reset_index(drop=False, inplace=True)
    return pt, new_cat_names


def print_missing(data,):
    '''
    Print missing colnames and counts
    '''
    if data.isna().any().sum() > 0:
        pprint(data.isna().sum()[data.isna().any()].to_dict())
    else:
        print('No missing values')

        
def change_ratio(current_value, previous_values, exclude_zeros=True, eps=1e-7,):
    '''
    Computes the ratio of change in current value in comparison with previous values
    Returns: (current_value - mean(previous_values)) / (stddev(previous_values) + eps)
    '''
    previous_values = np.array(previous_values)
    if exclude_zeros:
        nonzero_indices = np.nonzero(previous_values)
        previous_values = previous_values[nonzero_indices]
    return (current_value - np.mean(previous_values)) / (np.std(previous_values) + eps)


def neg_mse_to_rmse(neg_mse):
    '''
    Converts negative mse to rmse
    '''
    return np.abs(neg_mse) ** 0.5
    

def search_grid(model, X_data, y_data, search_space, metric, num_folds, metric_func=None):
    '''
    Searches in the provided space for a best metric
    Returns best model, train/test scores and best parameters
    '''
    grid = GridSearchCV(model, search_space, scoring=metric, cv=num_folds, 
                        n_jobs=-1, return_train_score=True)
    grid.fit(X_data, y_data)
    best_index = grid.best_index_
    val_score = grid.best_score_
    val_mean = grid.cv_results_['mean_test_score'].mean()
    train_score = grid.cv_results_['mean_train_score'][best_index]
    if metric_func is not None:
        train_score = metric_func(train_score)
        val_score = metric_func(val_score)
        val_mean = metric_func(val_mean)
    print('='*5, ' Grid Search Results ', '='*5)
    print('Best parameters are:')
    pprint(grid.best_params_)
    print('Best train score %.4f, validation %.4f (mean val %.4f)' % 
                          (train_score, val_score, val_mean))
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

    for i,(train_i,val_i) in enumerate(cv.split(train_data, train_labels)):
        if isinstance(train_data, (pd.core.frame.DataFrame, pd.core.series.Series)):
            X_train,y_train = train_data.iloc[train_i], train_labels.iloc[train_i]
            X_val,y_val = train_data.iloc[val_i], train_labels.iloc[val_i]
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


def reduce_dimensions(algorithm, data):
    '''
    Reduce dimensions of training data
    '''
    algorithm.fit(data,)
    print(f'Explained variance ratio {algorithm.explained_variance_ratio_.sum():.3f}')
    return algorithm, algorithm.transform(data)
