'''
A module to make certain data operations
'''
from pprint import pprint
from datetime import datetime as dt
import pandas as pd
import numpy as np


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


def balance_data(X_data, y_data, sampler, **kwargs):
    '''
    Balance input features and labels using samplers from imbalance-learn lib
    Return fitted sampler and balanced data
    '''
    try:
        from imblearn.over_sampling import SMOTE, KMeansSMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
    except ImportError as e:
        print(e)
        raise Exception('install "imblearn" lib first')

    sampler = sampler.lower()
    assert sampler in ['smote','over','under','comb','kmeans']
    if sampler == 'over':
        print('Random over-sampling with', kwargs)
        sampler = RandomOverSampler(**kwargs)
    elif sampler == 'smote':
        print('SMOTE over-sampling with', kwargs)
        sampler = SMOTE(**kwargs)
    elif sampler == 'under':
        print('Random under-sampling with', kwargs)
        sampler = RandomUnderSampler(**kwargs)
    elif sampler == 'comb':
        print('SMOTE upsample and ENN undersample with', kwargs)
        sampler = SMOTEENN(**kwargs)
    elif sampler == 'kmeans':
        print('K-Means SMOTE upsample with', kwargs)
        sampler = KMeansSMOTE(**kwargs)
    X_data,y_data = sampler.fit_resample(X_data, y_data)
    return sampler, X_data, y_data


def remove_nans(X, y):
    '''
    Remove NaNs at X_train set
    Impute NaNs with X_training means for X_test set
    '''
    is_nan = np.isnan(X.sum(1))
    X = X[is_nan == False]
    y = y[is_nan == False]
    print(f'Removed {is_nan.sum()} NaNs in the X set')
    return X, y


def replace_array_vals(replace_map, array, silent=True):
	'''
	Replace some array values with a new value
	'''
	assert isinstance(replace_map, dict) and isinstance(array, np.ndarray)
	if not silent: print('Initial labels:', set(array))
	for old,new in replace_map.items():
	    matches = array == old
	    array[matches] = new
	if not silent: print('New labels:', set(array))
	return array


def day_diff(from_date, to_date):
	''' Calculate number of days between two dates '''
	assert to_date >= from_date, 'from_date is later than to_date'
	if to_date == from_date:
	    return 0
	else:
	    return (to_date - from_date).days


def split_by_date(data, date_col, split_date):
    '''
    Splits dataset by dates:
        before split date is train set
        after split date is test set
    Split date is "%Y-%m-%d", date column is in UTC
    '''
    is_test = data.apply(lambda x: x[date_col] > pd.to_datetime(split_date, utc=True), 
                                        axis=1)
    train_data = data[~is_test]
    test_data = data[is_test]
    return train_data, test_data


def get_label_indices(labels):
    ''' Return unique labels and indices related to them as a list of tuples '''
    labels = np.array(labels)
    uniq_lab = np.unique(labels)
    res = []
    for lab in uniq_lab:
        indices = np.argwhere(labels == lab).squeeze(1)
        res.append((lab, indices))
    return res