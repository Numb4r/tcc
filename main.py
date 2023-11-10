import utils as u
from sklearn.model_selection import RepeatedKFold, cross_validate, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
# Utilizar GridSearchCV para pesquisa de hiperparâmetros e cross-validation
from sklearn.model_selection import GridSearchCV
import numpy as np
from itertools import product

prever = ["opt.runtime",]
data, results = u.load_files()
results = u.cleaning_columns(results, keep=prever)

params = [
    list(range(7)),  # Z_score_threshold
    list(range(1, 24)),  # selector_k
    list(range(1,))
]
for i in list(product(*params)):
    print(i)


def run(data, results, z_score_threshold: int, selector_n: int):
    X_train, X_test, y_train, y_test = u.split_train_test(data, results)
    print(len(X_train.keys()))
    # ==================================================================
    #  Remocao de outliers
    # ==================================================================
    indices = u.get_outliers(X_train, z_score_threshold=z_score_threshold)
    print(len(indices))
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
    model = DecisionTreeRegressor(random_state=0)
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
    # aacuracia 0.36
    #  k = 16
    #  acuracia de 38%

    kf = RepeatedKFold(n_splits=2, n_repeats=30, random_state=42)
    # kf = KFold(n_splits=2, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', scaler),
        ("selector", selector),
        ("model", model)
    ])
# ==================================================================
#  Montagem do Pipeline
# ==================================================================

# Utilizar GridSearchCV para pesquisa de hiperparâmetros e cross-validation
# param_grid = {'classifier__n_estimators': [50, 100, 200]}
# grid_search = GridSearchCV(
#     pipeline, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# # Obter os melhores parâmetros
# best_params = grid_search.best_params_


# ==================================================================
#  Configuracao do Cross-validation
# ==================================================================
['accuracy', 'precision', 'recall']
scoring = {
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2_score': 'r2'
}
# cv_results = cross_validate(grid_search.best_estimator_, X_train,
#                             y_train, cv=kf, scoring='accuracy', return_train_score=True)
# print(f'Acurácia no conjunto de treino: {cv_results["train_score"].mean()}')
# print(f'Acurácia na validação cruzada: {cv_results["test_score"].mean()}')

try:
    resultados = cross_validate(
        model, X_train, y_train, cv=kf, scoring=scoring)
except Exception as e:
    print(f"{e}")


# Exibir os resultados
for metric, values in resultados.items():
    if metric != 'fit_time' and metric != 'score_time':
        print(
            f'Métrica: {metric}, Média: {values.mean()}, Desvio Padrão: {values.std()}')


pipeline.fit(X_train, y_train)

# Faça previsões no conjunto de teste
previsoes_teste = pipeline.predict(X_test)

# Avalie o desempenho no conjunto de teste
acuracia_teste = pipeline.score(X_test, y_test)
# precision_teste = precision_score(y_test, previsoes_teste)
# recall_teste = recall_score(y_test, previsoes_teste)

print("Acurácia no conjunto de teste:", acuracia_teste)
# print("Precisão no conjunto de teste:", precision_teste)
# print("Revocação no conjunto de teste:", recall_teste)
