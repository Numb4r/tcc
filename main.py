from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from itertools import product
import pandas as pd
import utils as u


def run(data, results, z_score_threshold: int, selector_n: int, str_model: str, n_splits: int, prediction):
    output = {
        "input.z_score_threshold": z_score_threshold,
        "input.selector_n_params": selector_n,
        "input.model": str_model,
        "input.n_splits": n_splits,
        "input.selector": "SelectKBest",
        "predict": '_'.join(prediction)
    }
    X_train, X_test, y_train, y_test = u.split_train_test(data, results)

    # ==================================================================
    #  Remocao de outliers
    # ==================================================================
    indices = u.get_outliers(y_train, z_score_threshold=z_score_threshold)
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
    scaler = MinMaxScaler()
    model = u.get_model(str_model)
    selector = SelectKBest(score_func=f_regression,
                           k=selector_n)

    pipeline = Pipeline([
        ('scaler', scaler),
        ("selector", selector),
        ("regressor", model)
    ])
    # ==================================================================
    #  Montagem do Pipeline
    # ==================================================================

    # ==================================================================
    #  Configuracao do Cross-validation
    # ==================================================================

    scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}

    cv = KFold(n_splits=n_splits)
    param_grid = u.get_params(str_model)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='r2', refit='r2', n_jobs=-1)

    # Ajustando o modelo com validação cruzada
    grid_search.fit(X_train, y_train)

    # Avaliando o modelo no conjunto de teste
    y_pred = grid_search.predict(X_test)

    print(f"\nResultados para {model.__class__.__name__}:")
    print("Melhores parâmetros encontrados:")
    output["output.best_params_"] = grid_search.best_params_

    for param in param_grid.keys():
        output[f"output.{param}"] = grid_search.best_params_[param]
        print(f"Param {param}: {grid_search.best_params_[param]}")

    print("Relatório de Classificação:")
    print("Acurácia média durante a validação cruzada:")
    print(f"Acurácia média: {grid_search.best_score_:.4f}")

    output["output.explained_variance_score"] = explained_variance_score(
        y_test, y_pred)
    output["output.max_error"] = max_error(y_test, y_pred)
    output["output.mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
    output["output.mean_squared_error"] = mean_squared_error(y_test, y_pred)
    output["output.median_absolute_error"] = median_absolute_error(
        y_test, y_pred)
    output["output.r2_score"] = r2_score(y_test, y_pred)
    print(output)
    print("pred:",y_pred)
    print("test:",y_test)
    print("="*50)
    # new_params = {k.replace('regressor__', ''): v for k,
    #               v in grid_search.best_params_.items()}
    # output["output.mean_poisson_deviance"] = mean_poisson_deviance(
    #     y_test, y_pred)
    # output["output.mean_gamma_deviance"] = mean_gamma_deviance(y_test, y_pred)
    # output["output.mean_absolute_percentage_error"] = mean_absolute_percentage_error(
    #     y_test, y_pred)
    # output["output.d2_absolute_error_score"] = d2_absolute_error_score(
    #     y_test, y_pred)
    # output["output.d2_pinball_score"] = d2_pinball_score(y_test, y_pred)
    # output["output.d2_tweedie_score"] = d2_tweedie_score(y_test, y_pred)
#     scoring = {
#         'neg_mean_squared_error': 'neg_mean_squared_error',
#         'neg_mean_absolute_error': 'neg_mean_absolute_error',
#         'r2_score': 'r2',
#     }
#     resultados = cross_validate(
#         model, X_train, y_train, cv=kf, scoring=scoring)


# # Exibir os resultados
#     for metric, values in resultados.items():
#         if metric != 'fit_time'and metric != 'score_time':
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

    # prever = ["opt.runtime"]
    prever = ["opt.nodes"]
    data, results = u.load_files()
    results = u.cleaning_columns(results, keep=prever)
    data = u.normalize_columns(data)

    modelos = ['DecisionTree', 'RandomForest',
               'GradientBoosting']
    # ['MLP', 'Voting']
    params = [
        np.linspace(1, 3, 10),  # Z_score_threshold
        list(range(5, 24)),  # selector_k RFE,KBEST
        modelos,  # models
        list(range(2, 6)),  # n_split KFold
    ]
    output = []
    for i in list(product(*params)):
        output.append(
            run(data, results, *i, prediction=prever)
        )

    df = pd.DataFrame(output)
    file_name = f"output_{'_'.join(prever)}_{'_'.join(modelos)}"
    df.to_csv(f"{file_name}.csv")


if __name__ == "__main__":
    main()


# Parâmetros para o Decision Tree
