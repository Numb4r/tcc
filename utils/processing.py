from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def get_params(str_model):
    dt_params = {
        'criterion': ['mse', 'mae'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 3, 4, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    # Parâmetros para o Random Forest
    rf_params = {
        'n_estimators': [50, 100, 200],
        'criterion': ['mse', 'mae'],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Parâmetros para o Gradient Boost
    gb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 3, 4, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    # Parâmetros para o MLP Regressor
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    # Parâmetros para o Voting Regressor
    vr_params = {
        'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
    }

    # Dicionário de estruturas
    models_params = {
        'DecisionTree': dt_params,
        'RandomForest': rf_params,
        'GradientBoosting': gb_params,
        'MLP': mlp_params,
        'Voting': vr_params
    }
    return models_params[str_model]


def get_model(str_model):
    if str_model == 'DecisionTree':
        return DecisionTreeRegressor(random_state=0)
    elif str_model == 'RandomForest':
        return RandomForestRegressor(random_state=0)
    elif str_model == 'GradientBoosting':
        return GradientBoostingRegressor(random_state=0)
    elif str_model == 'MLP':
        return MLPRegressor(max_iter=1000, random_state=0)
    elif str_model == 'Voting':
        return VotingRegressor(estimators=[
            ('dt', DecisionTreeRegressor(random_state=0)),
            ('rf', RandomForestRegressor(random_state=0)),
            ('gb', GradientBoostingRegressor(random_state=0))
        ], n_jobs=-1)
