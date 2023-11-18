import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error, d2_absolute_error_score, d2_pinball_score, d2_tweedie_score
import utils as u


def get_r2_score_max(df: pd.DataFrame):
    r2 = np.array(df["output.r2_score"])
    m = np.argmax(r2)
    return m


def validate_score_max(data, results, config):

    X_train, X_test, y_train, y_test = u.split_train_test(data, results)
    ZSCORE = "input.z_score_threshold"
    z = config[ZSCORE]

    indices = u.get_outliers(y_train, z)
    X_train = X_train.drop(index=indices)
    y_train = y_train.drop(index=indices)
    y_train = np.array(y_train).reshape(1, -1)[0]
    str_model = config["input.model"]
    scaler = MinMaxScaler()
    model = u.get_model(str_model)
    max_depth = config["output.regressor__max_depth"]
    min_split = config["output.regressor__min_samples_split"]
    min_samples = config["output.regressor__min_samples_leaf"]
    model.set_params(max_depth=max_depth,
                     min_samples_split=min_split, min_samples_leaf=min_samples)
    selector_n = config["input.selector_n_params"]
    selector = SelectKBest(score_func=f_regression,
                           k=selector_n)

    pipeline = Pipeline([
        ('scaler', scaler),
        ("selector", selector),
        ("regressor", model)
    ])
    n_splits = config["input.n_splits"]
    cv = KFold(n_splits=n_splits)
    param_grid = u.get_params(str_model)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='r2', refit='r2', n_jobs=-1)

    # Ajustando o modelo com validação cruzada
    grid_search.fit(X_train, y_train)

    # Avaliando o modelo no conjunto de teste
    y_pred = grid_search.predict(X_test)

    print(
        f"output.explained_variance_score  {explained_variance_score(y_test, y_pred)}")
    print(f"output.max_error  {max_error(y_test, y_pred)}")
    print(f"output.mean_absolute_error  {mean_absolute_error(y_test, y_pred)}")
    print(f"output.mean_squared_error  {mean_squared_error(y_test, y_pred)}")
    print(
        f"output.median_absolute_error  {median_absolute_error(y_test, y_pred)}")
    print(f"output.r2_score  {r2_score(y_test, y_pred)}")


def main():
    output = pd.read_csv("output.csv")
    prever = ["opt.runtime"]
    prever = ["opt.nodes"]
    data, results = u.load_files()
    results = u.cleaning_columns(results, keep=prever)
    data = u.normalize_columns(data)
    index_max = get_r2_score_max(output)

    best_iteration = output.iloc[index_max]
    validate_score_max(data, results, best_iteration)


if __name__ == "__main__":
    main()
