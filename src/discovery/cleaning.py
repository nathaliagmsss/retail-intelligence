from __future__ import annotations

import pandas as pd


CAMPOS_OBRIGATORIOS: dict[str, list[str]] = {
    "compras": [
        "NF Número",
        "Emissão",
        "Ident.",
        "Descrição",
    ],
    "estoque": [
        "Código",
        "Descrição",
    ],
    "notas_venda": [
        "Núm./Série",
        "Emissão",
    ],
    "vendas_itens": [
        "Nota",
        "Data",
        "Código",
        "Descrição do item",
    ],
}


def remover_linhas_totalmente_vazias(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove linhas em que todas as colunas estão vazias ou nulas.
    """
    df = dataframe.copy()

    colunas_texto = df.select_dtypes(
        include=["object", "string"],
    ).columns

    for coluna in colunas_texto:
        df[coluna] = df[coluna].replace(
            r"^\s*$",
            pd.NA,
            regex=True,
        )

    return df.dropna(how="all").reset_index(drop=True)


def remover_linhas_sem_campos_obrigatorios(
    dataframe: pd.DataFrame,
    fonte: str,
) -> pd.DataFrame:
    """
    Remove linhas que não possuem nenhum dos campos essenciais
    da respectiva fonte.

    O uso de `how="all"` evita excluir registros que tenham apenas um
    campo pontualmente ausente.
    """
    campos = CAMPOS_OBRIGATORIOS.get(fonte, [])

    campos_existentes = [
        campo
        for campo in campos
        if campo in dataframe.columns
    ]

    if not campos_existentes:
        return dataframe.copy()

    return (
        dataframe
        .dropna(
            subset=campos_existentes,
            how="all",
        )
        .reset_index(drop=True)
    )


def limpar_dataframe_discovery(
    dataframe: pd.DataFrame,
    fonte: str,
) -> pd.DataFrame:
    """
    Executa somente limpezas estruturais seguras.

    Nesta etapa não são corrigidos decimais, tipos, datas ou códigos.
    """
    df = remover_linhas_totalmente_vazias(dataframe)

    df = remover_linhas_sem_campos_obrigatorios(
        dataframe=df,
        fonte=fonte,
    )

    return df.reset_index(drop=True)