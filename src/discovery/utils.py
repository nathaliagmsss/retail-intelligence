from __future__ import annotations

from typing import Any

import pandas as pd


MAX_EXEMPLOS = 5
MAX_TAMANHO_EXEMPLO = 100


def limpar_texto_exemplo(valor: Any) -> str:
    """
    Converte um valor em texto curto para uso nos relatórios.

    O valor original não é alterado.
    """
    if pd.isna(valor):
        return ""

    texto = " ".join(str(valor).strip().split())

    if len(texto) > MAX_TAMANHO_EXEMPLO:
        return texto[: MAX_TAMANHO_EXEMPLO - 3] + "..."

    return texto


def obter_exemplos(
    serie: pd.Series,
    limite: int = MAX_EXEMPLOS,
) -> str:
    """
    Retorna exemplos distintos e não nulos de uma coluna.
    """
    exemplos: list[str] = []
    vistos: set[str] = set()

    for valor in serie.dropna():
        texto = limpar_texto_exemplo(valor)

        if not texto or texto in vistos:
            continue

        exemplos.append(texto)
        vistos.add(texto)

        if len(exemplos) >= limite:
            break

    return " | ".join(exemplos)


def contar_strings_vazias(serie: pd.Series) -> int:
    """
    Conta strings vazias ou compostas somente por espaços.

    Valores nulos não entram nessa contagem.git status

    """
    if not (
        pd.api.types.is_object_dtype(serie)
        or pd.api.types.is_string_dtype(serie)
    ):
        return 0

    valores = serie.dropna().astype("string").str.strip()

    return int(valores.eq("").sum())