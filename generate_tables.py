import numpy as np
import pandas as pd

nodes = pd.read_csv(
    "final/output_opt.nodes_DecisionTree_RandomForest_GradientBoosting.csv")
runtime = pd.read_csv(
    "final/output_opt.runtime_DecisionTree_RandomForest_GradientBoosting.csv")


linha_zeros = pd.DataFrame(
    np.zeros((1, runtime.shape[1])), columns=runtime.columns)
runtime_d = {
    "best": {
        "r2": 0.4018419391,
        "mae": 792,
        "mse": 3347010.168
    },
    "worst": {
        "r2": -0.13,
        "mae": 1240,
        "mse":  6397283
    }
}
nodes_d = {
    "best": {
        "r2": 0.92,
        "mae": 53045,
        "mse": 9792419717
    },
    "worst": {
        "r2": -1.42,
        "mae": 219532,
        "mse": 314475480863
    }
}
for d in [runtime_d, nodes_d]:
    for t in ["best", "worst"]:
        if d == runtime_d:
            str_n = "runtime"
            results = runtime
        else:
            str_n = "nodes"
            results = nodes
        # print(t)
        if t == "best":
            r2_df = results[results["output.r2_score"] >= d[t]["r2"]]
            mae_df = results[results["output.mean_absolute_error"]
                             <= d[t]["mae"]]
            mse_df = results[results["output.mean_squared_error"]
                             <= d[t]["mse"]]
        else:
            r2_df = results[results["output.r2_score"] <= d[t]["r2"]]
            mae_df = results[results["output.mean_absolute_error"]
                             >= d[t]["mae"]]
            mse_df = results[results["output.mean_squared_error"]
                             >= d[t]["mse"]]

        final = dfs_concatenados = pd.concat(
            [r2_df, linha_zeros, mae_df, linha_zeros, mse_df], ignore_index=True)
        final = final.drop(["input.num_outliers", "predict", "input.selector", "output.best_params_", "Unnamed: 0",
                            "output.median_absolute_error", "output.max_error", "output.explained_variance_score"], axis=1)
        final = final.round(3)

        final.to_csv(f"resultados_{str_n}_{t}.csv")

        final = final.apply(lambda x: x.astype(str) + ' &')

        final_texto = final.to_string(index=False)

        # Substituir vírgulas por "&"
        final_texto = final_texto.replace(',', '&')

        # Adicionar "\\ ao final de cada linha"
        linhas = final_texto.split('\n')
        linhas_formatadas = [linha + ' \\\\' for linha in linhas]

        # Juntar as linhas formatadas de volta em um único texto
        texto_final = '\n'.join(linhas_formatadas)

        with open(f"resultados_{str_n}_{t}.txt", 'w') as arquivo:
            # Escrever o conteúdo da variável no arquivo
            arquivo.write(texto_final)
