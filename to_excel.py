import pandas as pd
import sys
# Ler o arquivo CSV
arquivo_csv = sys.argv[1]
dados_csv = pd.read_csv(arquivo_csv)
# print(arquivo_csv[:-4])
# Especificar o nome do arquivo XLSX de saída
arquivo_xlsx = f'{arquivo_csv[:-4]}.xlsx'

# Salvar os dados no arquivo XLSX
dados_csv.to_excel(arquivo_xlsx, index=False)

print(f'O arquivo CSV foi convertido com sucesso para {arquivo_xlsx}.')
