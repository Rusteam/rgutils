# ~*~ coding: utf-8 ~*~
'''
A module with classes and functions for 
build a machine learning pipeline
from loading features and labels to 
to submitting results to Mlflow Tracking
board
'''
import os, sys
from copy import deepcopy
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlflow
# custom modules
from src.rgutils.data import replace_array_vals, remove_nans, balance_data
from src.rgutils.ml import reduce_dimensions, cross_validation, f_score_macro, scale_features
from src.rgutils.nlp import vectorize_text


class BuildModel:
    '''
    Create a pipeline to build a model
    Implement these steps:
        1. Load feature datasets (order matters)
        2. Load corresponding labels (order matters)
        3. Customize labels: {remap labels, select only target labels}
        4. Remove rows from training with missing values or outliers
            OR impute data
        5. Processing steps as Tf-Idf or PCA
        6. Combine features for training or ensemble predictions or build a metalearner
        7. Cross-validated training
        8. Evaluate on test data: individually for each test set or mixed
        9. Submit results to the mlflow tracking board 
    '''
    process_options = ['decomposer','vectorizer','scaler']
    subset_options = ['features','concat']
    train_data_options = ['concat','ensemble','stack']
    
    def __init__(self, feature_dict, label_list):
        # make sure input features and labels match in names
        l_names = [n for n,_ in label_list]
        for _,features in feature_dict.items():
            f_names = [n for n,_ in features]
            assert f_names == l_names, f"{f_names} are different from labels {l_names}"
        self.features = {}
        for name,features in feature_dict.items():
            self.features[name] = []
            for set_name,path in features:
                self.features[name].append([set_name, self._load_data(path)])
        self.labels = {name:self._load_data(path) for name,path in label_list}
        self.train_params = {}
        self._update_general_params()
        self.set2index = {name:c for c,(name,_) in enumerate(label_list)}
        self.index2set = {v:k for k,v in self.set2index.items()}
        self.num_sets = len(self.set2index)
        self._check_sizes()
        

    def _update_general_params(self):
        ''' Get sizes of sets and unique of labels '''
        self.train_params.update({f"{set_name}_size": len(set_labels) \
                                for set_name,set_labels in self.labels.items()})
        classes = set()
        _ = [classes.update(set(set_labels)) for set_labels in self.labels.values()]
        self.train_params.update({
            'classes': list(classes),
            'features': list(self.features.keys()),
            })

        
    @staticmethod
    def _load_data(path):
        assert os.path.exists(path), f'{path} does not exist'
        file_extension = path.split('.')[-1]
        if file_extension == 'csv':
            data = pd.read_csv(path)
        elif file_extension == 'npy':
            data = np.load(path, allow_pickle=True)
        elif file_extension == 'txt':
            with open(path, 'r') as f:
                data = np.array(f.read().strip('\n').split('\n'))
        else:
            raise Exception(f'Format {file_extension} not supported for {path}')
        print('Loaded data of shape', data.shape, path)
        return data
    

    def _check_sizes(self, subset='features'):
        '''
        Test if features and labels have equal number of examples
        Test if train and test have equal number of features
        Test if labels have only 1 dimension
        ------
        Subset argument tells which set of features to test {features, concat}
        '''
        def x_equals_y(set_name, set_data, name):
            ''' check if size of X and y equal for each set '''
            assert len(set_data) == len(self.labels[set_name]), \
                    f'{name} {set_name} set size differ from size of labels'
        
        def equal_num_features(features, name):
            ''' check if sets have equal number of features '''
            if len(features) == 0:
                raise Exception('No features provided')
            elif len(features) == 1:
                return
            else:
                set_names = [n for n,_ in features]
                first_index = self.set2index[set_names[0]]
                if features[0][1].ndim == 1:
                    return
                for set_name in set_names[1:]:
                    second_index = self.set2index[set_name]
                    assert features[first_index][1].shape[1] == features[second_index][1].shape[1], \
                            (f'{set_names[self.index2set[first_index]]} not equal to '
                            f'{set_names[self.index2set[second_index]]} features {name}')
                    
        assert subset in self.subset_options
        if subset == 'features':
            for name,features in self.features.items():
                for set_name,set_data in features:
                    x_equals_y(set_name, set_data, name)
                equal_num_features(features, name)
        elif subset == 'concat':
            assert isinstance(self.concat, dict) and len(self.concat) > 0
            feature_num = self.concat[self.index2set[0]].shape[1] # feature num
            for set_name, set_data in self.concat.items():
                x_equals_y(set_name, set_data, 'concat')
                assert set_data.shape[1] == feature_num, \
                    f'Concat {set_name} not equal to feature number {feature_num}'
            
        for name,array in self.labels.items():
            assert array.ndim == 1, f'Label {name} does not have 1 dimension'
        print('DATA LOOKS GOOD after size checkup')
    

    def relabel_targets(self, mapping=None, include=None):
        '''
        Relabel target variable 
        Select only given list of target classes if specified
        '''
        assert mapping is not None or include is not None
        print()
        self.old_labels = deepcopy(self.labels)
        if mapping:
            print("RELABELING TARGETS")
            self.labels = {name:replace_array_vals(mapping, lab) for name,lab in self.labels.items()}
        if include:
            print()
            print('SELECTING TARGET CLASSES')
            assert len(include) >= 2
            is_target_fn = lambda x: x in include
            # Go over each set of labels
            # get target indices and select those indices
            # in labels and respective features
            for set_name,labels in self.labels.items():
                target_i = np.array(list(map(is_target_fn, labels)))
                self.labels[set_name] = labels[target_i]
                self.old_labels[set_name] = self.old_labels[set_name][target_i]
                index = self.set2index[set_name]
                for name,features in self.features.items():
                    self.features[name][index][1] = features[index][1][target_i]
                print(f'Selected {sum(target_i)} rows at {name}')
        self._check_sizes()
        self._update_general_params()
    

    def process_features(self, process_steps):
        '''
        Apply processing to input features such as
        text vectorization or decomposition of features
        -----
        Params:
            process_steps is a dict with a list of tuples for each feauture
                        as {feature_name: [(step_name, algorithm, optional: args),...], ...} 
                        in the order provided
        '''
        print()
        print("PROCESSING FEATURES")
        p_keys = list(process_steps.keys())
        f_keys = list(self.features.keys())
        for k in p_keys: assert k in f_keys
        for name,step in process_steps.items():
            for step_conf in step:
                assert step_conf[0] in self.process_options, \
                    f"Process step {step_conf[0]} has to be among {self.process_options}"
                transformed = self._process_step(name, *step_conf)
                self.features[name] = [[self.index2set[i], t] \
                                        for i,t in zip(range(self.num_sets), transformed)]
        self._check_sizes()
        self.train_params.update(process_steps)
    

    def _process_step(self, feature_name, process_type, process_algo, process_args={}):
        ''' Call text vectorizer or reduce dimensions '''
        # order sets inside feature according to how they were passed to label_list
        # for all sets, get set_name for index, then index of that set
        # then for the feature take the second element of that index 
        input_data = [self.features[feature_name][self.set2index[self.index2set[i]]][1] \
                      for i in range(self.num_sets)]
        if process_type == 'vectorizer':
            _,transformed = vectorize_text(*input_data, vectorizer=process_algo, **process_args)
        elif process_type == 'decomposer':
            _,transformed = reduce_dimensions(*input_data, decomposer=process_algo, **process_args)
        elif process_type == 'scaler':
            _,transformed = scale_features(*input_data, scaler=process_algo, **process_args)
        return transformed
    

    def concat_features(self):
        ''' Concatenate sets of features to create one set '''
        print()
        print('CONCATENATING FEATURES')
        self.concat = {}
        self.concat_names = []
        for c,(name,features) in enumerate(self.features.items()):
            for set_name,set_index in self.set2index.items():
                set_data = features[set_index][1]
                if c == 0:
                    self.concat.update({set_name: set_data})
                else:
                    self.concat[set_name] = np.concatenate([self.concat[set_name], set_data], axis=1)
            # concat feature names
            if hasattr(set_data, 'columns'):
                f_names = set_data.columns.values
            else:
                f_names = [f"{name}_{i}" for i in range(set_data.shape[1])]
            self.concat_names.extend(f_names)
        for set_name,set_data in self.concat.items():
            print(f'{set_name} has shape {set_data.shape} after concatenation')
        self._check_sizes('concat')


    def resample_data(self, sampler, sampling_configs={}, subset='features', ):
        '''
        Resample data by over-sampling, under-sampling or combining
        Use imbalance-learn lib
        '''
        assert subset in self.subset_options
        # resample train set
        set_name = self.index2set[0]
        set_index = self.set2index[set_name]
        if subset == 'concat':
            X_train = self.concat[set_name]
            y_train = self.labels[set_name]
            _,X_train,y_train = balance_data(X_train, y_train, sampler, **sampling_configs)
            self.concat[set_name] = deepcopy(X_train)
            self.labels[set_name] = deepcopy(y_train)
        elif subset == 'ensemble':
            for feature_name,features in self.features.items():
                X_train = features[set_index][1]
                y_train = self.labels[set_name]
                _,X_train,y_train = balance_data(X_train, y_train, sampler, **sampling_configs)
                self.features[feature_name][set_index][1] = deepcopy(X_train)
                if labels:
                    assert sum(labels != y_train) == 0, f"Different y labels resampled for ensemble"
                labels = deepcopy(y_train)
            self.labels[set_name] = deepcopy(labels)
        self._check_sizes(subset=subset)
        self._update_general_params()
        self.train_params.update({'sampler': (sampler, sampling_configs, subset)})
    

    def build_models(self, estimators, metric, num_folds, train_data='concat', k_best=3):
        ''' Train a list of models and return best of them '''
        assert train_data in self.train_data_options
        mode_fn = lambda x: stats.mode(x).mode.squeeze()
        
        def train_model(estimator):
            '''
            Cross-validate a model on a set and test it on a holdout set
            '''            
            cv_models,cv_scores = \
                    cross_validation(X_train, y_train, estimator, num_folds,
                                     metric, model_name='cv', stratified=True)
            print(f'Cross-validation scores:')
            d = pd.DataFrame(cv_scores)
            mean_train,std_train = d["train"].mean(), d["train"].std()
            mean_val,std_val = d["val"].mean(), d["val"].std()
            print(f'Train: {mean_train:.3f} +- {std_train:.4f}')
            print(f'Validation: {mean_val:.3f} +- {std_val:.4f}')
            predictions = np.empty((len(y_test), len(cv_models)), dtype='<U20', )
            for i,(name,model) in enumerate(cv_models.items()):
                predictions[:,i] = model.predict(X_test)
            final = np.array(list(map(mode_fn, predictions))) 
            print()
            print('Classification report:')
            print(classification_report(y_test, final))   
            print()
            print('Test set confusion matrix:')
            print(pd.crosstab(y_test, final))
            results = dict(
                test_metric = round(metric(y_test, final), 4),
                test_accuracy = round(accuracy_score(y_test, final), 4),
                crossval_metric_mean = round(mean_val, 4),
                crossval_metric_std = round(std_val, 4),
                )
            return results, final
        
        def run_estimators():
            ''' Run all estimators through data '''
            y_predicted = {}
            for algo in estimators:
                print()
                print('TRAINING', str(algo.__class__.__name__))
                try:
                    results,cv_predicted = train_model(algo)
                    params = deepcopy(self.train_params)
                    algo_name = str(algo.__class__.__name__)
                    params.update({'algorithm': algo_name,
                                   'train_data': train_data,
                                   'metric': str(metric.__name__)})                    
                    if train_data == 'concat':
                        self.runs.append({'params': params, 'results': results})
                    id_str = '_'.join([algo_name, train_data if train_data == 'concat' else set_name])
                    y_predicted[id_str] = cv_predicted
                except ValueError as e:
                    print(e)
            return y_predicted
        
        self.runs = []
        print()
        print('RUN TRAINING AS', train_data.upper())
        y_train = self.labels[self.index2set[0]]
        y_test = self.labels[self.index2set[1]]        
        # train models on concatenated features
        if train_data == 'concat':
            assert isinstance(self.concat, dict) and len(self.concat) > 0
            X_train = self.concat[self.index2set[0]]
            X_test = self.concat[self.index2set[1]]
            _ = run_estimators()
        # iterate over features and train models on all of them
        elif train_data == 'ensemble':
            # train models for each set and get cv predictions
            test_predictions = {}
            for set_name,set_data in self.features.items():
                X_train = set_data[self.set2index[self.index2set[0]]][1]
                X_test = set_data[self.set2index[self.index2set[1]]][1]
                y_predicted = run_estimators()
                test_predictions.update(y_predicted)
            test_predictions = pd.DataFrame(test_predictions)
            # get weak scores for submission
            results = {}
            print()
            for pred in test_predictions.columns:
                weak_score = metric(y_test, test_predictions[pred])
                results[pred] = round(weak_score, 4)
                print(f'Weak {pred} score: {weak_score:.4f}')
            # get final prediction and score
            best_models = sorted(results, key=lambda x: results[x], reverse=True)[:k_best]
            self.final_prediction = test_predictions[best_models].mode(1)[0]
            ensemble_score = metric(y_test, self.final_prediction)
            results['test_metric'] = round(ensemble_score, 4)
            print(f'Ensemble score: {ensemble_score:.4f}', best_models)
            # add params and results for submission
            params = deepcopy(self.train_params)
            params.update({'algorithm': [str(a.__class__.__name__) for a in estimators],
                           'train_data': train_data,
                           'metric': str(metric.__name__),
                           'best_models': best_models})
            self.runs.append({'params': params, 'results': results})
        # iterate over features, train models on all of them and train a meta-learner
        elif train_data == 'stack':
            raise NotImplementedError
    

    def save_results(self, experiment_name):
        ''' Save results with Mlflow Tracking '''
        print()
        print('SUBMITTING RESULTS TO MLFLOW')
        mlflow.set_experiment(experiment_name)
        for run in self.runs:
            with mlflow.start_run():
                mlflow.log_params(run['params'])
                mlflow.log_metrics(run['results'])
