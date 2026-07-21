from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.discovery.analisar_html import (
    extrair_textos_celulas,
    ler_sopa_html,
    linha_valida_compras,
    selecionar_tabela_principal,
)
from src.discovery.cleaning import limpar_dataframe_discovery


COLUNAS_COMPRAS = [
    "NF Número",
    "Série",
    "Mod.",
    "Emissão",
    "Entrada",
    "Ident.",
    "Descrição",
    "Quant.",
    "Custo Compra",
]


def extrair_dataframe_compras(
    caminho: Path,
) -> pd.DataFrame:
    """
    Extrai a estrutura lógica do relatório de compras.

    O FastReport utiliza 25 colunas físicas de layout, mas cada registro
    possui nove campos reais.
    """
    sopa = ler_sopa_html(caminho)
    registros: list[list[str]] = []

    for linha in sopa.find_all("tr"):
        textos = extrair_textos_celulas(linha)

        if linha_valida_compras(textos):
            registros.append(textos)

    if not registros:
        raise ValueError(
            "Nenhuma linha válida encontrada no relatório de compras."
        )

    return pd.DataFrame(
        registros,
        columns=COLUNAS_COMPRAS,
    )


def extrair_dataframe_padrao(
    caminho: Path,
    fonte: str,
) -> pd.DataFrame:
    """
    Extrai relatórios que podem ser interpretados pelo pandas.

    A tabela principal é escolhida pela compatibilidade dos cabeçalhos.
    """
    tabelas = pd.read_html(caminho)

    if not tabelas:
        raise ValueError("Nenhuma tabela HTML encontrada.")

    _, tabela, pontuacao = selecionar_tabela_principal(
        tabelas=tabelas,
        fonte=fonte,
    )

    if pontuacao == 0:
        raise ValueError(
            f"Nenhum cabeçalho esperado reconhecido para '{fonte}'."
        )

    return tabela.copy()


def extrair_dataframe(
    caminho: Path,
    fonte: str,
) -> pd.DataFrame:
    """
    Extrai e limpa estruturalmente o DataFrame principal da fonte.
    """
    if fonte == "compras":
        dataframe = extrair_dataframe_compras(caminho)
    else:
        dataframe = extrair_dataframe_padrao(
            caminho=caminho,
            fonte=fonte,
        )

    return limpar_dataframe_discovery(
        dataframe=dataframe,
        fonte=fonte,
    )