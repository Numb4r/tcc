import utils as u
from sklearn.model_selection import RepeatedKFold, cross_validate, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
# Utilizar GridSearchCV para pesquisa de hiperparâmetros e cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from itertools import product
import pandas as pd
prever = ["opt.runtime",]
data, results = u.load_files()
results = u.cleaning_columns(results, keep=prever)


def run(data, results, z_score_threshold: int, selector_n: int, n_splits: int, model):
    output = {
        "input.z_score_threshold": z_score_threshold,
        "input.selector_n_params": selector_n,
        "input.n_splits_Kf": n_splits
    }

    X_train, X_test, y_train, y_test = u.split_train_test(data, results)
    # print(len(X_train.keys()))
    # ==================================================================
    #  Remocao de outliers
    # ==================================================================
    indices = u.get_outliers(X_train, z_score_threshold=z_score_threshold)
    output["input.num_outliers"] = len(indices)
    X_train = X_train.drop(index=indices)
    y_train = y_train.drop(index=indices)
    # ==================================================================
    #  Remocao de outliers
    # ==================================================================
    y_train = np.array(y_train).reshape(1, -1)[0]

    # ==================================================================
    #  Montagem do Pipeline
    # ==================================================================
    selector_n = min(selector_n, len(indices))
    scaler = MinMaxScaler()

    output["input.model"] = model.__class__.__name__

    # selector = RFE(estimator=model, n_features_to_select=selector_n)  # Ruim ?
    selector = SelectKBest(score_func=f_regression,
                           k=selector_n)  # 9 10 11 12
    # z score = 3
    # k = 9
    # acuraria de 28%

    # z score = 4
    # k = 15
    # acuraria de 30%

    # Z score =6
    #  k = 15
    #  acuracia 0.36
    #  k = 16
    #  acuracia de 38%

    kf = RepeatedKFold(n_splits=n_splits, n_repeats=30, random_state=42)
    # kf = KFold(n_splits=2, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', scaler),
        ("selector", selector),
        ("model", model)
    ])
    # ==================================================================
    #  Montagem do Pipeline
    # ==================================================================

    # ==================================================================
    #  Configuracao do Cross-validation
    # ==================================================================
    ['accuracy', 'precision', 'recall']
    scoring = {
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'r2_score': 'r2'
    }

    resultados = cross_validate(
        model, X_train, y_train, cv=kf, scoring=scoring)


# Exibir os resultados
    for metric, values in resultados.items():
        if metric != 'fit_time' and metric != 'score_time':
            output[f"output.{metric}"] = values.mean()
            output[f"output.{metric}.standard_deviation"] = values.std()
            print(
                f'Métrica: {metric}, Média: {values.mean()}, Desvio Padrão: {values.std()}')

    pipeline.fit(X_train, y_train)

    # Faça previsões no conjunto de teste
    previsoes_teste = pipeline.predict(X_test)

    # Avalie o desempenho no conjunto de teste
    acuracia_teste = pipeline.score(X_test, y_test)
    output["output.accuracy"] = acuracia_teste

    print("Acurácia no conjunto de teste:", acuracia_teste)
    return output


params = [
    list(range(1, 10)),  # Z_score_threshold
    list(range(1, 24)),  # selector_k RFE,KBEST
    list(range(2, 3)),  # n splits KFOLD
    [DecisionTreeRegressor(random_state=0), RandomForestRegressor(
        random_state=0), GradientBoostingRegressor(random_state=0)]  # models
]
output = []
for i in list(product(*params)):
    output.append(
        run(data, results, *i)
    )
df = pd.DataFrame(output)
df.to_csv("output.csv")
