from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate, KFold


from sklearn.feature_selection import SelectKBest, f_regression, RFE


from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error, d2_absolute_error_score, d2_pinball_score, d2_tweedie_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from itertools import product
import pandas as pd
import utils as u


def run(data, results, z_score_threshold: int, selector_n: int, model, index_selector: int):
    # print("PARAMGRID:", param_grid)

    output = {
        "input.z_score_threshold": z_score_threshold,
        "input.selector_n_params": selector_n,
        "input.model": model.__class__.__name__
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
    l_selector = [RFE(estimator=model, n_features_to_select=selector_n), SelectKBest(score_func=f_regression,
                                                                                     k=selector_n)]
    selector = l_selector[index_selector]
    output["input.selector"] = selector.__class__.__name__

    # kf = RepeatedKFold(n_splits=n_splits, n_repeats=30, random_state=42)
    # kf = KFold(n_splits=2, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', scaler),
        ("selector", selector),
        ("regressor", DecisionTreeRegressor())
    ])
    # ==================================================================
    #  Montagem do Pipeline
    # ==================================================================

    # ==================================================================
    #  Configuracao do Cross-validation
    # ==================================================================
    scoring = {
        'neg_mean_squared_error': 'neg_mean_squared_error',
        # 'neg_mean_absolute_error': 'neg_mean_absolute_error',
        # 'r2_score': 'r2',
        # 'explained_variance': 'explained_variance',
        # 'max_error': 'max_error',

    }
    scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}

    cv = 5
    param_grid = {
        # Ajuste o parâmetro dentro do regressor no pipeline
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='r2', refit='r2', n_jobs=-1)

    # Ajustando o modelo com validação cruzada
    grid_search.fit(X_train, y_train)

    # Avaliando o modelo no conjunto de teste
    y_pred = grid_search.predict(X_test)

    print(f"\nResultados para {model.__class__.__name__}:")
    print("Melhores parâmetros encontrados:")
    print(grid_search.best_params_)

    output["output.best_params_"] = grid_search.best_params_
    print("Relatório de Classificação:")
    # grid_search.best_params_
    # print(classification_report(y_test, y_pred))
    print("Acurácia média durante a validação cruzada:")
    print(f"Acurácia média: {grid_search.best_score_:.4f}")
    print("="*50)
    # a = DecisionTreeRegressor()
    new_params = {k.replace('regressor__', ''): v for k,
                  v in grid_search.best_params_.items()}
    model.set_params(**new_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    output["output.explained_variance_score"] = explained_variance_score(
        y_test, y_pred)
    output["output.max_error"] = max_error(y_test, y_pred)
    output["output.mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
    output["output.mean_squared_error"] = mean_squared_error(y_test, y_pred)
    output["output.mean_squared_error"] = mean_squared_error(y_test, y_pred)
    output["output.mean_squared_log_error"] = mean_squared_log_error(
        y_test, y_pred)
    output["output.median_absolute_error"] = median_absolute_error(
        y_test, y_pred)
    output["output.r2_score"] = r2_score(y_test, y_pred)
    output["output.mean_poisson_deviance"] = mean_poisson_deviance(
        y_test, y_pred)
    output["output.mean_gamma_deviance"] = mean_gamma_deviance(y_test, y_pred)
    output["output.mean_absolute_percentage_error"] = mean_absolute_percentage_error(
        y_test, y_pred)
    output["output.d2_absolute_error_score"] = d2_absolute_error_score(
        y_test, y_pred)
    output["output.d2_pinball_score"] = d2_pinball_score(y_test, y_pred)
    output["output.d2_tweedie_score"] = d2_tweedie_score(y_test, y_pred)
#     scoring = {
#         'neg_mean_squared_error': 'neg_mean_squared_error',
#         'neg_mean_absolute_error': 'neg_mean_absolute_error',
#         'r2_score': 'r2',
#     }
#     resultados = cross_validate(
#         model, X_train, y_train, cv=kf, scoring=scoring)


# # Exibir os resultados
#     for metric, values in resultados.items():
#         if metric != 'fit_time' and metric != 'score_time':
#             output[f"output.{metric}"] = values.mean()
#             output[f"output.{metric}.standard_deviation"] = values.std()
#             print(
#                 f'Métrica: {metric}, Média: {values.mean()}, Desvio Padrão: {values.std()}')

#     pipeline.fit(X_train, y_train)

#     # Faça previsões no conjunto de teste
#     previsoes_teste = pipeline.predict(X_test)

#     # Avalie o desempenho no conjunto de teste
#     acuracia_teste = pipeline.score(X_test, y_test)
#     output["output.pipeline_accuracy"] = acuracia_teste

#     print("Acurácia no conjunto de teste:", acuracia_teste)
    return output


def main():
    prever = ["opt.runtime",]
    data, results = u.load_files()
    results = u.cleaning_columns(results, keep=prever)
    params = [
        list(range(1, 10)),  # Z_score_threshold
        list(range(1, 24)),  # selector_k RFE,KBEST
        # param_grid
        [DecisionTreeRegressor(random_state=0), RandomForestRegressor(
            random_state=0), GradientBoostingRegressor(random_state=0)],  # models
        [0, 1],  # index list selector method
    ]
    output = []
    for i in list(product(*params)):
        output.append(
            run(data, results, *i)
        )
    df = pd.DataFrame(output)
    df.to_csv("output.csv")


if __name__ == "__main__":
    main()
