
# AlertaSP — Documentação dos Dados

## 1. Objetivo

Documentar os datasets reais utilizados no desenvolvimento do modelo de Machine Learning do AlertaSP, incluindo fontes, períodos, variáveis e transformações.

A pesquisa utiliza dados de precipitação, temperatura e umidade, relacionados a registros históricos de alagamentos e inundações no município de São Paulo.

## 2. Fonte meteorológica — INMET

**Instituição:** Instituto Nacional de Meteorologia (INMET)

**Site oficial:**
https://portal.inmet.gov.br/dadoshistoricos

**Arquivos anuais:**
https://portal.inmet.gov.br/uploads/dadoshistoricos/2022.zip

Os scripts utilizam o mesmo padrão de endereço para baixar os anos necessários.

### Estações utilizadas

| Código | Estação |
|---|---|
| A701 | São Paulo — Mirante |
| A771 | São Paulo — Interlagos |

### Variáveis extraídas

- Data e hora UTC
- Precipitação horária (mm)
- Temperatura do ar (°C)
- Umidade relativa do ar (%)
- Identificador da estação

### Tratamentos realizados

1. Leitura dos arquivos CSV contidos nos ZIPs anuais.
2. Conversão de números com vírgula decimal.
3. Identificação de valores ausentes ou inválidos.
4. Conversão de UTC para America/Sao_Paulo.
5. Remoção de registros duplicados por estação e horário.
6. Agregação das medições por dia.
7. Cálculo de precipitação diária apenas quando existem 24 medições válidas.

Os dados meteorológicos de 2021 são utilizados para complementar a virada para 2022.

## 3. Fonte de alagamentos — GeoSampa

**Órgãos de origem:** Prefeitura de São Paulo e Defesa Civil.

Catálogo oficial:
https://metadados.geosampa.prefeitura.sp.gov.br/geonetwork/srv/api/records/432c06b1-03a1-4b1f-9210-423e1b58e869

### Dataset efetivamente utilizado

Repositório público de consolidação:
https://github.com/23216886alunounivesp/sp-alagamentos-inundacoes-defesa-civil

Arquivo:
`alagamentos_e_inundacoes_sp.csv`

URL de download utilizada no script 03:

https://raw.githubusercontent.com/23216886alunounivesp/sp-alagamentos-inundacoes-defesa-civil/main/alagamentos_e_inundacoes_sp.csv

**Total do arquivo consultado:** 12.487 registros.

**Período declarado:** 2013–2026.

### Colunas originais

| Coluna | Descrição |
|---|---|
| X | Coordenada UTM horizontal |
| Y | Coordenada UTM vertical |
| data | Data da ocorrência |
| ocorrencia | Tipo: ALAGAMENTO ou INUNDACAO |
| subprefeit | Subprefeitura responsável |
| name | Identificador/metadado de origem, quando disponível |

Referência espacial declarada: SIRGAS 2000 / UTM zona 23S (EPSG:31983).

### Tratamentos realizados

1. Padronização de datas que possuíam formatos diferentes.
2. Normalização de textos e categorias.
3. Conversão das coordenadas para valores numéricos.
4. Filtragem dos eventos ALAGAMENTO e INUNDACAO.
5. Agregação da quantidade de ocorrências por dia.
6. Cruzamento com os dados meteorológicos pela data.

O modelo considera ambos os tipos de ocorrência na classe positiva.

### Limitações da fonte

O arquivo utilizado é uma consolidação publicada por terceiros, baseada em registros oficiais. Não foi concluída uma auditoria linha a linha contra os arquivos originais do GeoSampa.

Além disso, o conjunto não fornece horário completo para todos os eventos. Por esse motivo, a modelagem utiliza resolução diária.

A ausência de registros em determinado dia não significa necessariamente que não houve alagamentos.

## 4. Registros utilizados por ano

| Ano | Dias com registro | Ocorrências |
|---|---:|---:|
| 2022 | 101 | 390 |
| 2023 | 123 | 803 |
| 2024 | 100 | 451 |
| 2025 | 85 | 663 |
| 2026 (parcial) | 69 | 448 |

Os resultados de 2026 são provisórios. No processamento realizado, a última ocorrência do CSV possui data de 03/07/2026, enquanto os dados meteorológicos alcançam outubro de 2026.

Não está comprovado que a fonte de ocorrências possui cobertura completa até 03/07/2026.

## 5. Base consolidada

Arquivo gerado:

`dados/processados/base_diaria_2022_2026.csv`

O arquivo combina informações meteorológicas diárias com a quantidade de ocorrências registradas.

### Variável-alvo

`alvo_registro_amanha`

- 1: existe pelo menos um registro de alagamento ou inundação no município no dia seguinte.
- 0: não existe registro correspondente na base utilizada para o dia seguinte.

O valor 0 representa ausência de registro, não ausência física comprovada de alagamento.

## 6. Features do modelo

As 12 variáveis de entrada são:

1. chuva_mm_a701
2. temperatura_media_c_a701
3. umidade_media_pct_a701
4. chuva_acumulada_3d_a701
5. chuva_acumulada_7d_a701
6. chuva_mm_a771
7. temperatura_media_c_a771
8. umidade_media_pct_a771
9. chuva_acumulada_3d_a771
10. chuva_acumulada_7d_a771
11. dia_ano_seno
12. dia_ano_cosseno

A sazonalidade é codificada por seno e cosseno para representar a natureza cíclica dos dias do ano.

As precipitações acumuladas são calculadas somente com o dia atual e os dias anteriores.

## 7. Separação dos datasets

| Conjunto | Amostras | Positivas |
|---|---:|---:|
| Treino 2022–2024 | 1.093 | 324 |
| Validação 2025 | 364 | 85 |
| Teste provisório 2026 | 183 | 69 |

Os dados de 2026 não participaram do treinamento nem das métricas finais apresentadas.

A última data de cada período é retirada quando não existe uma data futura do mesmo ano para compor a variável-alvo.

## 8. Valores ausentes

Os dados ausentes são mantidos como NaN durante a preparação.

No treinamento, utiliza-se SimpleImputer(strategy="median"), ajustado apenas aos dados do respectivo conjunto de treinamento.

Isso evita calcular valores de substituição utilizando informações da validação.

## 9. Reproduzir os datasets

Dentro da pasta `ML`, com o ambiente virtual ativo:

```bash
python scripts/01_inspecionar_inmet.py
python scripts/02_preparar_inmet.py
python scripts/03_preparar_alagamentos.py
python scripts/04_cruzar_dados.py
python scripts/05_analisar_dados.py
python scripts/06_consolidar_historico.py
python scripts/07_preparar_features.py
```

Os scripts 08 e 09 utilizam os dados preparados para realizar o treinamento e as avaliações.

## 10. Reprodutibilidade

Os resultados documentados correspondem aos arquivos consultados durante o desenvolvimento em novembro de 2026.

As fontes externas podem ser atualizadas. Para garantir resultados exatamente iguais, devem ser preservados os arquivos originais, seus hashes e as versões das bibliotecas.

Nenhum dataset sintético foi utilizado no treinamento.
