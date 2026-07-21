from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.discovery.analisar_html import (
    INPUT_DIR,
    OUTPUT_DIR,
    extrair_metadados_arquivo,
    localizar_arquivos_html,
)
from src.discovery.extractors import extrair_dataframe


SUMMARY_PATH = OUTPUT_DIR / "relacionamentos_resumo.csv"
CODES_MISSING_FROM_STOCK_PATH = (
    OUTPUT_DIR / "codigos_ausentes_estoque.csv"
)

CODES_WITH_EMPTY_CATEGORY_PATH = (
    OUTPUT_DIR / "codigos_com_categoria_vazia.csv"
)
NOTE_DIFFERENCES_PATH = OUTPUT_DIR / "notas_com_diferenca.csv"
NOTE_COVERAGE_PATH = OUTPUT_DIR / "cobertura_notas.csv"

TOLERANCIA_MONETARIA = 0.01

MetodoValorItens = Literal[
    "valor_linha",
    "preco_unitario_vezes_quantidade",
]


def normalizar_identificador(valor: Any) -> str | None:
    """
    Normaliza códigos e números de notas como texto.

    Exemplos:
        10228.0 -> "10228"
        " 07677 " -> "07677"
        NaN -> None

    Observação:
        zeros à esquerda já perdidos pelo pandas não podem ser
        reconstruídos por esta função.
    """
    if pd.isna(valor):
        return None

    texto = str(valor).strip()

    if not texto:
        return None

    texto = re.sub(r"\.0$", "", texto)

    return texto or None


def extrair_numero_nota(valor: Any) -> str | None:
    """
    Extrai o número principal da nota.

    Exemplos:
        "149597/1" -> "149597"
        149597.0 -> "149597"
    """
    identificador = normalizar_identificador(valor)

    if identificador is None:
        return None

    numero = identificador.split("/", maxsplit=1)[0].strip()

    return numero or None


def converter_numero_bruto(
    serie: pd.Series,
    divisor: float = 1.0,
) -> pd.Series:
    """
    Converte uma coluna numérica lida pelo pandas.

    Nos relatórios de vendas, notas e estoque, o pandas removeu a
    vírgula decimal. Por isso, valores como 85,00 aparecem como 8500.

    O divisor deve ser informado explicitamente pela regra da fonte.
    """
    numerico = pd.to_numeric(
        serie,
        errors="coerce",
    )

    return numerico / divisor


def converter_data(
    serie: pd.Series,
) -> pd.Series:
    """
    Converte datas no formato brasileiro.
    """
    return pd.to_datetime(
        serie,
        format="%d/%m/%Y",
        errors="coerce",
    )


def obter_arquivos_por_fonte(
    fonte: str,
) -> list[Path]:
    """
    Retorna os arquivos da loja Pilar pertencentes a uma fonte.
    """
    arquivos = localizar_arquivos_html(INPUT_DIR)

    selecionados: list[Path] = []

    for caminho in arquivos:
        metadados = extrair_metadados_arquivo(caminho)

        if metadados["fonte"] == fonte:
            selecionados.append(caminho)

    return sorted(selecionados)


def carregar_fonte(
    fonte: str,
) -> pd.DataFrame:
    """
    Carrega e concatena todos os arquivos de uma fonte.

    Acrescenta:
        - arquivo_origem;
        - ano_arquivo.
    """
    arquivos = obter_arquivos_por_fonte(fonte)

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado para a fonte '{fonte}'."
        )

    dataframes: list[pd.DataFrame] = []

    for indice, caminho in enumerate(arquivos, start=1):
        metadados = extrair_metadados_arquivo(caminho)

        print(
            f"[{indice}/{len(arquivos)}] "
            f"Carregando {fonte}: {metadados['arquivo']}"
        )

        dataframe = extrair_dataframe(
            caminho=caminho,
            fonte=fonte,
        ).copy()

        dataframe["arquivo_origem"] = metadados["arquivo"]
        dataframe["ano_arquivo"] = metadados["ano"]

        dataframes.append(dataframe)

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


def preparar_vendas_itens(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Padroniza apenas os campos necessários à análise de relacionamentos.
    """
    df = dataframe.copy()

    df["nota_normalizada"] = df["Nota"].map(
        extrair_numero_nota
    )

    df["codigo_normalizado"] = df["Código"].map(
        normalizar_identificador
    )

    df["data_normalizada"] = converter_data(
        df["Data"]
    )

    df["ano_venda"] = (
        df["data_normalizada"]
        .dt.year
        .astype("Int64")
    )

    # 100 representa 1,00 no HTML interpretado pelo pandas.
    df["quantidade_normalizada"] = converter_numero_bruto(
        df["Quantidade"],
        divisor=100.0,
    )

    # 8500 representa R$ 85,00.
    df["valor_vendido_normalizado"] = converter_numero_bruto(
        df["Vendido por"],
        divisor=100.0,
    )

    df["custo_compra_normalizado"] = converter_numero_bruto(
        df["Custo compra"],
        divisor=100.0,
    )

    df["descricao_normalizada"] = (
        df["Descrição do item"]
        .astype("string")
        .str.strip()
    )

    return df


def preparar_estoque(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara o snapshot atual de estoque como fonte de categoria.
    """
    df = dataframe.copy()

    df["codigo_normalizado"] = df["Código"].map(
        normalizar_identificador
    )

    df["descricao_estoque"] = (
        df["Descrição"]
        .astype("string")
        .str.strip()
    )

    df["categoria"] = (
    df["Grupo"]
    .astype("string")
    .str.strip()
    .replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
        }
    )
)

    return df


def preparar_notas(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Padroniza os campos necessários do relatório resumido de notas.
    """
    df = dataframe.copy()

    df["nota_normalizada"] = df["Núm./Série"].map(
        extrair_numero_nota
    )

    df["data_normalizada"] = converter_data(
        df["Emissão"]
    )

    df["ano_venda"] = (
        df["data_normalizada"]
        .dt.year
        .astype("Int64")
    )

    df["total_nota_normalizado"] = converter_numero_bruto(
        df["TOTAL"],
        divisor=100.0,
    )

    df["mercadorias_normalizado"] = converter_numero_bruto(
        df["Mercadorias"],
        divisor=100.0,
    )

    if "Desc./Acr.Prod." in df.columns:
        df["desconto_acrescimo_normalizado"] = (
            converter_numero_bruto(
                df["Desc./Acr.Prod."],
                divisor=100.0,
            )
        )
    else:
        df["desconto_acrescimo_normalizado"] = pd.NA

    return df


def criar_chave_nota(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Cria uma chave composta por ano e número da nota.

    O ano é incluído para evitar colisões caso a numeração se repita
    em períodos diferentes.
    """
    ano = dataframe["ano_venda"].astype("string")
    nota = dataframe["nota_normalizada"].astype("string")

    chave = ano + "::" + nota

    chave = chave.mask(
        dataframe["ano_venda"].isna()
        | dataframe["nota_normalizada"].isna()
    )

    return chave


def consolidar_estoque_por_codigo(
    estoque: pd.DataFrame,
) -> pd.DataFrame:
    """
    Garante uma linha por código no cadastro derivado do estoque.

    Se houver repetição, preserva a primeira descrição e categoria
    não nulas e registra quantas linhas existiam.
    """
    validos = estoque[
        estoque["codigo_normalizado"].notna()
    ].copy()

    return (
        validos
        .groupby(
            "codigo_normalizado",
            as_index=False,
        )
        .agg(
            descricao_estoque=(
                "descricao_estoque",
                "first",
            ),
            categoria=(
                "categoria",
                "first",
            ),
            ocorrencias_no_estoque=(
                "codigo_normalizado",
                "size",
            ),
        )
    )

def resumir_codigos_vendidos(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume códigos de produtos vendidos para análise de cobertura.
    """
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "codigo_normalizado",
                "descricao_venda_exemplo",
                "primeira_venda",
                "ultima_venda",
                "anos_com_venda",
                "linhas_vendidas",
                "quantidade_vendida",
                "valor_vendido",
            ]
        )

    return (
        dataframe
        .groupby(
            "codigo_normalizado",
            as_index=False,
        )
        .agg(
            descricao_venda_exemplo=(
                "descricao_normalizada",
                "first",
            ),
            primeira_venda=(
                "data_normalizada",
                "min",
            ),
            ultima_venda=(
                "data_normalizada",
                "max",
            ),
            anos_com_venda=(
                "ano_venda",
                lambda valores: " | ".join(
                    str(int(valor))
                    for valor in sorted(
                        valores.dropna().unique()
                    )
                ),
            ),
            linhas_vendidas=(
                "codigo_normalizado",
                "size",
            ),
            quantidade_vendida=(
                "quantidade_normalizada",
                "sum",
            ),
            valor_vendido=(
                "valor_vendido_normalizado",
                "sum",
            ),
        )
        .sort_values(
            by=[
                "valor_vendido",
                "linhas_vendidas",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

def analisar_cobertura_categorias(
    vendas: pd.DataFrame,
    estoque: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Relaciona vendas e estoque pelo código do produto.

    Separa:
    1. códigos vendidos ausentes do estoque atual;
    2. códigos encontrados no estoque, mas sem categoria preenchida.

    Compras não participam deste relacionamento.
    """
    cadastro = consolidar_estoque_por_codigo(
        estoque
    )

    # Marcador explícito para distinguir ausência no estoque
    # de categoria vazia.
    cadastro["encontrado_no_estoque"] = True

    vendas_enriquecidas = vendas.merge(
        cadastro,
        how="left",
        on="codigo_normalizado",
        validate="many_to_one",
    )

    vendas_enriquecidas["encontrado_no_estoque"] = (
        vendas_enriquecidas["encontrado_no_estoque"]
        .eq(True)
    )

    codigos_ausentes = vendas_enriquecidas[
        vendas_enriquecidas[
            "codigo_normalizado"
        ].notna()
        & ~vendas_enriquecidas[
            "encontrado_no_estoque"
        ]
    ].copy()

    codigos_categoria_vazia = vendas_enriquecidas[
        vendas_enriquecidas[
            "codigo_normalizado"
        ].notna()
        & vendas_enriquecidas[
            "encontrado_no_estoque"
        ]
        & vendas_enriquecidas[
            "categoria"
        ].isna()
    ].copy()

    resumo_ausentes = resumir_codigos_vendidos(
        codigos_ausentes
    )

    resumo_categoria_vazia = resumir_codigos_vendidos(
        codigos_categoria_vazia
    )

    return (
        vendas_enriquecidas,
        resumo_ausentes,
        resumo_categoria_vazia,
    )


def consolidar_valores_itens_por_nota(
    vendas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida os itens usando duas hipóteses:

    1. `Vendido por` já representa o total da linha;
    2. `Vendido por` representa preço unitário e deve ser multiplicado
       pela quantidade.
    """
    df = vendas.copy()

    df["chave_nota"] = criar_chave_nota(df)

    df["valor_hipotese_linha"] = (
        df["valor_vendido_normalizado"]
    )

    df["valor_hipotese_unitario"] = (
        df["valor_vendido_normalizado"]
        * df["quantidade_normalizada"]
    )

    validos = df[
        df["chave_nota"].notna()
    ].copy()

    return (
        validos
        .groupby(
            [
                "chave_nota",
                "ano_venda",
                "nota_normalizada",
            ],
            as_index=False,
        )
        .agg(
            quantidade_linhas_itens=(
                "chave_nota",
                "size",
            ),
            quantidade_total=(
                "quantidade_normalizada",
                "sum",
            ),
            valor_itens_hipotese_linha=(
                "valor_hipotese_linha",
                "sum",
            ),
            valor_itens_hipotese_unitario=(
                "valor_hipotese_unitario",
                "sum",
            ),
        )
    )


def consolidar_notas(
    notas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Garante uma linha por ano e número de nota.
    """
    df = notas.copy()
    df["chave_nota"] = criar_chave_nota(df)

    validos = df[
        df["chave_nota"].notna()
    ].copy()

    return (
        validos
        .groupby(
            [
                "chave_nota",
                "ano_venda",
                "nota_normalizada",
            ],
            as_index=False,
        )
        .agg(
            total_nota=(
                "total_nota_normalizado",
                "sum",
            ),
            quantidade_registros_nota=(
                "chave_nota",
                "size",
            ),
            data_nota=(
                "data_normalizada",
                "min",
            ),
        )
    )


def inferir_metodo_valor_itens(
    notas_com_itens: pd.DataFrame,
) -> tuple[MetodoValorItens, dict[str, int]]:
    """
    Determina qual interpretação de `Vendido por` produz mais
    conciliações exatas com o total das notas.
    """
    comuns = notas_com_itens[
        notas_com_itens["total_nota"].notna()
        & notas_com_itens[
            "valor_itens_hipotese_linha"
        ].notna()
        & notas_com_itens[
            "valor_itens_hipotese_unitario"
        ].notna()
    ].copy()

    diferenca_linha = (
        comuns["valor_itens_hipotese_linha"]
        - comuns["total_nota"]
    ).abs()

    diferenca_unitario = (
        comuns["valor_itens_hipotese_unitario"]
        - comuns["total_nota"]
    ).abs()

    correspondencias_linha = int(
        diferenca_linha.le(
            TOLERANCIA_MONETARIA
        ).sum()
    )

    correspondencias_unitario = int(
        diferenca_unitario.le(
            TOLERANCIA_MONETARIA
        ).sum()
    )

    if correspondencias_unitario > correspondencias_linha:
        metodo: MetodoValorItens = (
            "preco_unitario_vezes_quantidade"
        )
    else:
        metodo = "valor_linha"

    metricas = {
        "correspondencias_valor_linha": (
            correspondencias_linha
        ),
        "correspondencias_preco_unitario": (
            correspondencias_unitario
        ),
        "notas_comuns_avaliadas": len(comuns),
    }

    return metodo, metricas


def analisar_relacao_notas(
    vendas: pd.DataFrame,
    notas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MetodoValorItens, dict[str, int]]:
    """
    Faz a conciliação entre notas resumidas e vendas por item.
    """
    itens_por_nota = consolidar_valores_itens_por_nota(
        vendas
    )
    notas_consolidadas = consolidar_notas(
        notas
    )

    comparacao = notas_consolidadas.merge(
        itens_por_nota,
        how="outer",
        on=[
            "chave_nota",
            "ano_venda",
            "nota_normalizada",
        ],
        indicator=True,
        validate="one_to_one",
    )

    metodo, metricas = inferir_metodo_valor_itens(
        comparacao
    )

    if metodo == "valor_linha":
        comparacao["valor_itens_escolhido"] = (
            comparacao[
                "valor_itens_hipotese_linha"
            ]
        )
    else:
        comparacao["valor_itens_escolhido"] = (
            comparacao[
                "valor_itens_hipotese_unitario"
            ]
        )

    comparacao["diferenca_valor"] = (
        comparacao["valor_itens_escolhido"]
        - comparacao["total_nota"]
    )

    comparacao["diferenca_absoluta"] = (
        comparacao["diferenca_valor"].abs()
    )

    comparacao["status_relacionamento"] = comparacao[
        "_merge"
    ].map(
        {
            "both": "presente_em_ambos",
            "left_only": "somente_notas",
            "right_only": "somente_vendas_itens",
        }
    )

    comparacao["valor_confere"] = (
        comparacao["_merge"].eq("both")
        & comparacao["diferenca_absoluta"].le(
            TOLERANCIA_MONETARIA
        )
    )

    divergencias = comparacao[
        comparacao["status_relacionamento"].ne(
            "presente_em_ambos"
        )
        | ~comparacao["valor_confere"]
    ].copy()

    colunas_saida = [
        "ano_venda",
        "nota_normalizada",
        "data_nota",
        "status_relacionamento",
        "quantidade_registros_nota",
        "quantidade_linhas_itens",
        "quantidade_total",
        "total_nota",
        "valor_itens_hipotese_linha",
        "valor_itens_hipotese_unitario",
        "valor_itens_escolhido",
        "diferenca_valor",
        "diferenca_absoluta",
        "valor_confere",
    ]

    comparacao = (
        comparacao[colunas_saida]
        .sort_values(
            by=[
                "ano_venda",
                "nota_normalizada",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    divergencias = (
        divergencias[colunas_saida]
        .sort_values(
            by=[
                "ano_venda",
                "diferenca_absoluta",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return (
        comparacao,
        divergencias,
        metodo,
        metricas,
    )


def percentual(
    numerador: int | float,
    denominador: int | float,
) -> float:
    """
    Calcula percentual com proteção contra divisão por zero.
    """
    if not denominador:
        return 0.0

    return round(
        (numerador / denominador) * 100,
        4,
    )


def criar_resumo_relacionamentos(
    vendas: pd.DataFrame,
    vendas_enriquecidas: pd.DataFrame,
    estoque: pd.DataFrame,
    codigos_ausentes_estoque: pd.DataFrame,
    codigos_categoria_vazia: pd.DataFrame,
    comparacao_notas: pd.DataFrame,
    metodo_valor_itens: MetodoValorItens,
    metricas_metodo: dict[str, int],
) -> pd.DataFrame:
    """
    Consolida os principais indicadores da Issue 4.
    """
    registros: list[dict[str, Any]] = []

    codigos_vendidos = {
        valor
        for valor in vendas["codigo_normalizado"].dropna()
    }

    codigos_estoque = {
        valor
        for valor in estoque["codigo_normalizado"].dropna()
    }

    codigos_encontrados = codigos_vendidos.intersection(
        codigos_estoque
    )

    linhas_vendas_validas = int(
        vendas_enriquecidas[
            "codigo_normalizado"
        ].notna().sum()
    )

    linhas_com_categoria = int(
        vendas_enriquecidas[
            "categoria"
        ].notna().sum()
    )

    registros.extend(
        [
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "codigos_distintos_vendidos",
                "valor": len(codigos_vendidos),
                "observacao": "",
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "codigos_distintos_no_estoque",
                "valor": len(codigos_estoque),
                "observacao": "Snapshot atual do estoque.",
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "codigos_vendidos_encontrados_no_estoque",
                "valor": len(codigos_encontrados),
                "observacao": "",
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "cobertura_codigos_percentual",
                "valor": percentual(
                    len(codigos_encontrados),
                    len(codigos_vendidos),
                ),
                "observacao": (
                    "Percentual dos códigos distintos vendidos "
                    "que aparecem no estoque atual."
                ),
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "linhas_vendas_com_categoria",
                "valor": linhas_com_categoria,
                "observacao": "",
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "cobertura_linhas_categoria_percentual",
                "valor": percentual(
                    linhas_com_categoria,
                    linhas_vendas_validas,
                ),
                "observacao": (
                    "Percentual das linhas de venda que podem "
                    "receber categoria pelo código."
                ),
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "codigos_vendidos_ausentes_estoque",
                "valor": len(codigos_ausentes_estoque),
                "observacao": (
                    "Códigos históricos vendidos que não aparecem "
                    "no snapshot atual de estoque."
                ),
            },
            {
                "relacionamento": "vendas_itens_x_estoque",
                "metrica": "codigos_encontrados_sem_categoria",
                "valor": len(codigos_categoria_vazia),
                "observacao": (
                    "Códigos encontrados no estoque atual cujo campo "
                    "Grupo/categoria está vazio."
                ),
            },
        ]
    )

    notas_em_ambos = int(
        comparacao_notas[
            "status_relacionamento"
        ].eq("presente_em_ambos").sum()
    )

    somente_notas = int(
        comparacao_notas[
            "status_relacionamento"
        ].eq("somente_notas").sum()
    )

    somente_itens = int(
        comparacao_notas[
            "status_relacionamento"
        ].eq("somente_vendas_itens").sum()
    )

    notas_com_valor_correto = int(
        comparacao_notas[
            "valor_confere"
        ].sum()
    )

    registros.extend(
        [
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "notas_presentes_em_ambos",
                "valor": notas_em_ambos,
                "observacao": "",
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "notas_somente_no_relatorio_resumido",
                "valor": somente_notas,
                "observacao": "",
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "notas_somente_em_vendas_itens",
                "valor": somente_itens,
                "observacao": (
                    "Pode incluir 2026, pois ainda não existe "
                    "relatório resumido desse ano."
                ),
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "cobertura_notas_percentual",
                "valor": percentual(
                    notas_em_ambos,
                    notas_em_ambos + somente_notas,
                ),
                "observacao": (
                    "Cobertura das notas resumidas por registros "
                    "de vendas por item."
                ),
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "notas_com_valor_conciliado",
                "valor": notas_com_valor_correto,
                "observacao": "",
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "conciliacao_valor_percentual",
                "valor": percentual(
                    notas_com_valor_correto,
                    notas_em_ambos,
                ),
                "observacao": (
                    "Percentual das notas presentes em ambos "
                    "cujos valores conferem."
                ),
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "metodo_valor_itens_escolhido",
                "valor": metodo_valor_itens,
                "observacao": (
                    "Método escolhido automaticamente pela "
                    "quantidade de conciliações exatas."
                ),
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "matches_hipotese_valor_linha",
                "valor": metricas_metodo[
                    "correspondencias_valor_linha"
                ],
                "observacao": "",
            },
            {
                "relacionamento": "notas_venda_x_vendas_itens",
                "metrica": "matches_hipotese_preco_unitario",
                "valor": metricas_metodo[
                    "correspondencias_preco_unitario"
                ],
                "observacao": "",
            },
            {
                "relacionamento": "compras_x_demais_fontes",
                "metrica": "integracao_automatica",
                "valor": False,
                "observacao": (
                    "Compras utiliza outro sistema e outro código "
                    "de produto. Nenhum join automático foi feito."
                ),
            },
        ]
    )

    return pd.DataFrame(registros)


def salvar_resultados(
    resumo: pd.DataFrame,
    codigos_ausentes_estoque: pd.DataFrame,
    codigos_categoria_vazia: pd.DataFrame,
    notas_com_diferenca: pd.DataFrame,
    cobertura_notas: pd.DataFrame,
) -> None:
    """
    Salva os resultados da análise de relacionamentos.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumo.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    codigos_ausentes_estoque.to_csv(
        CODES_MISSING_FROM_STOCK_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    codigos_categoria_vazia.to_csv(
        CODES_WITH_EMPTY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    notas_com_diferenca.to_csv(
        NOTE_DIFFERENCES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    cobertura_notas.to_csv(
        NOTE_COVERAGE_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def exibir_resumo_terminal(
    resumo: pd.DataFrame,
    codigos_ausentes_estoque: pd.DataFrame,
    codigos_categoria_vazia: pd.DataFrame,
    notas_com_diferenca: pd.DataFrame,
    metodo_valor_itens: MetodoValorItens,
) -> None:
    """
    Exibe os principais resultados da execução.
    """
    print("\nAnálise de relacionamentos concluída.")
    print("Loja analisada: pilar")
    print(
        "Método escolhido para o valor dos itens: "
        f"{metodo_valor_itens}"
    )
    print(
    "Códigos vendidos ausentes do estoque atual: "
    f"{len(codigos_ausentes_estoque)}"
)

    print(
        "Códigos encontrados no estoque sem categoria: "
        f"{len(codigos_categoria_vazia)}"
    )
    print(
        "Notas ausentes ou com diferença: "
        f"{len(notas_com_diferenca)}"
    )
    print(f"Resumo salvo em: {SUMMARY_PATH}")
    print(
    "Códigos ausentes do estoque salvos em: "
    f"{CODES_MISSING_FROM_STOCK_PATH}"
)

    print(
        "Códigos com categoria vazia salvos em: "
        f"{CODES_WITH_EMPTY_CATEGORY_PATH}"
    )
    print(
        "Notas com diferença salvas em: "
        f"{NOTE_DIFFERENCES_PATH}"
    )
    print(
        "Cobertura completa das notas salva em: "
        f"{NOTE_COVERAGE_PATH}"
    )

    print("\nPrincipais métricas:")

    for _, linha in resumo.iterrows():
        if linha["metrica"] in {
            "cobertura_codigos_percentual",
            "cobertura_linhas_categoria_percentual",
            "cobertura_notas_percentual",
            "conciliacao_valor_percentual",
        }:
            print(
                f"- {linha['metrica']}: "
                f"{linha['valor']}%"
            )


def main() -> None:
    """
    Executa a Issue 4: avaliação de chaves e relacionamentos.
    """
    print("Carregando vendas por item...")
    vendas_brutas = carregar_fonte(
        "vendas_itens"
    )

    print("\nCarregando estoque...")
    estoque_bruto = carregar_fonte(
        "estoque"
    )

    print("\nCarregando notas de venda...")
    notas_brutas = carregar_fonte(
        "notas_venda"
    )

    vendas = preparar_vendas_itens(
        vendas_brutas
    )
    estoque = preparar_estoque(
        estoque_bruto
    )
    notas = preparar_notas(
        notas_brutas
    )

    (
        vendas_enriquecidas,
        codigos_ausentes_estoque,
        codigos_categoria_vazia,
    ) = analisar_cobertura_categorias(
        vendas=vendas,
        estoque=estoque,
    )

    (
        cobertura_notas,
        notas_com_diferenca,
        metodo_valor_itens,
        metricas_metodo,
    ) = analisar_relacao_notas(
        vendas=vendas,
        notas=notas,
    )

    resumo = criar_resumo_relacionamentos(
        vendas=vendas,
        vendas_enriquecidas=vendas_enriquecidas,
        estoque=estoque,
        codigos_ausentes_estoque=codigos_ausentes_estoque,
        codigos_categoria_vazia=codigos_categoria_vazia,
        comparacao_notas=cobertura_notas,
        metodo_valor_itens=metodo_valor_itens,
        metricas_metodo=metricas_metodo,
    )

    salvar_resultados(
        resumo=resumo,
        codigos_ausentes_estoque=codigos_ausentes_estoque,
        codigos_categoria_vazia=codigos_categoria_vazia,
        notas_com_diferenca=notas_com_diferenca,
        cobertura_notas=cobertura_notas,
    )

    exibir_resumo_terminal(
        resumo=resumo,
        codigos_ausentes_estoque=codigos_ausentes_estoque,
        codigos_categoria_vazia=codigos_categoria_vazia,
        notas_com_diferenca=notas_com_diferenca,
        metodo_valor_itens=metodo_valor_itens,
    )


if __name__ == "__main__":
    main()