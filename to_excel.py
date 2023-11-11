import pandas as pd

# Ler o arquivo CSV
arquivo_csv = 'output.csv'
dados_csv = pd.read_csv(arquivo_csv)

# Especificar o nome do arquivo XLSX de saída
arquivo_xlsx = 'saida.xlsx'

# Salvar os dados no arquivo XLSX
dados_csv.to_excel(arquivo_xlsx, index=False)

print(f'O arquivo CSV foi convertido com sucesso para {arquivo_xlsx}.')
