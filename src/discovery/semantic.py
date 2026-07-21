from __future__ import annotations

import pandas as pd


IDENTIFICADORES = {
    "código",
    "codigo",
    "código barras",
    "codigo barras",
    "nota",
    "nf número",
    "nf numero",
    "núm./série",
    "num./serie",
    "ident.",
    "série",
    "serie",
    "mod.",
}

DATAS = {
    "data",
    "emissão",
    "emissao",
    "entrada",
    "últ.compra",
    "ult.compra",
    "últ.venda",
    "ult.venda",
    "data saída",
    "data saida",
}

HORAS = {
    "hora saída",
    "hora saida",
}

MONETARIOS = {
    "vendido por",
    "custo compra",
    "preço em r$",
    "preco em r$",
    "mercadorias",
    "desc./acr.prod.",
    "total",
    "qtd x preço",
    "qtd x preco",
}

QUANTIDADES = {
    "quantidade",
    "quant.",
}


def inferir_tipo_semantico(
    nome_coluna: str,
    serie: pd.Series,
) -> str:
    """
    Sugere o papel semântico de uma coluna.

    Essa classificação é apenas uma hipótese de Data Discovery.
    """
    nome = nome_coluna.casefold().strip()

    if nome in IDENTIFICADORES:
        return "identificador"

    if nome in DATAS:
        return "data"

    if nome in HORAS:
        return "hora"

    if nome in MONETARIOS:
        return "monetário"

    if nome in QUANTIDADES:
        return "quantidade"

    if pd.api.types.is_numeric_dtype(serie):
        return "numérico"

    return "texto"