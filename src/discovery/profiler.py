"""Responsável por responder: Como é esse DataFrame?"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.discovery.analisar_html import (
    INPUT_DIR,
    OUTPUT_DIR,
    extrair_metadados_arquivo,
    localizar_arquivos_html,
)
from src.discovery.extractors import extrair_dataframe
from src.discovery.semantic import inferir_tipo_semantico
from src.discovery.utils import (
    contar_strings_vazias,
    obter_exemplos,
)


PROFILE_PATH = OUTPUT_DIR / "perfil_colunas.csv"
ERRORS_PATH = OUTPUT_DIR / "erros_perfil.csv"


def criar_perfil_coluna(
    serie: pd.Series,
    metadados: dict[str, Any],
    posicao: int,
    duplicatas_arquivo: int,
) -> dict[str, Any]:
    """
    Gera as métricas de profiling de uma coluna.
    """
    total_linhas = len(serie)
    nulos = int(serie.isna().sum())
    vazios = contar_strings_vazias(serie)
    preenchidos = total_linhas - nulos - vazios

    serie_texto = (
        serie
        .dropna()
        .astype("string")
        .str.strip()
    )

    valores_unicos = int(
        serie_texto[serie_texto.ne("")].nunique()
    )

    percentual_nulos = (
        round((nulos / total_linhas) * 100, 4)
        if total_linhas
        else 0.0
    )

    percentual_vazios = (
        round((vazios / total_linhas) * 100, 4)
        if total_linhas
        else 0.0
    )

    percentual_unicos = (
        round((valores_unicos / preenchidos) * 100, 4)
        if preenchidos
        else 0.0
    )

    nome_coluna = str(serie.name)

    return {
        "arquivo": metadados["arquivo"],
        "caminho_relativo": metadados["caminho_relativo"],
        "loja": metadados["loja"],
        "fonte": metadados["fonte"],
        "ano": metadados["ano"],
        "posicao_coluna": posicao,
        "coluna": nome_coluna,
        "tipo_pandas": str(serie.dtype),
        "tipo_semantico_sugerido": inferir_tipo_semantico(
            nome_coluna=nome_coluna,
            serie=serie,
        ),
        "total_linhas": total_linhas,
        "valores_preenchidos": preenchidos,
        "nulos": nulos,
        "nulos_percentual": percentual_nulos,
        "strings_vazias": vazios,
        "strings_vazias_percentual": percentual_vazios,
        "valores_unicos": valores_unicos,
        "valores_unicos_percentual": percentual_unicos,
        "duplicatas_completas_arquivo": duplicatas_arquivo,
        "exemplos": obter_exemplos(serie),
    }


def criar_perfil_dataframe(
    dataframe: pd.DataFrame,
    metadados: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Gera o perfil de todas as colunas de um DataFrame.
    """
    duplicatas_arquivo = int(
        dataframe.duplicated().sum()
    )

    registros: list[dict[str, Any]] = []

    for posicao, coluna in enumerate(dataframe.columns):
        serie = dataframe.iloc[:, posicao].copy()
        serie.name = coluna

        registros.append(
            criar_perfil_coluna(
                serie=serie,
                metadados=metadados,
                posicao=posicao,
                duplicatas_arquivo=duplicatas_arquivo,
            )
        )

    return registros


def gerar_perfil(
    arquivos: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o profiling de todos os arquivos.

    Returns:
        DataFrame com perfil das colunas e DataFrame com erros.
    """
    perfis: list[dict[str, Any]] = []
    erros: list[dict[str, Any]] = []

    total = len(arquivos)

    for indice, caminho in enumerate(
        arquivos,
        start=1,
    ):
        metadados = extrair_metadados_arquivo(caminho)
        fonte = str(metadados["fonte"])

        print(
            f"[{indice}/{total}] "
            f"Perfilando: {metadados['caminho_relativo']}"
        )

        try:
            dataframe = extrair_dataframe(
                caminho=caminho,
                fonte=fonte,
            )

            perfis.extend(
                criar_perfil_dataframe(
                    dataframe=dataframe,
                    metadados=metadados,
                )
            )

        except Exception as exc:
            erro = {
                "arquivo": metadados["arquivo"],
                "caminho_relativo": metadados["caminho_relativo"],
                "loja": metadados["loja"],
                "fonte": metadados["fonte"],
                "ano": metadados["ano"],
                "tipo_erro": type(exc).__name__,
                "erro": str(exc),
            }

            erros.append(erro)

            print(
                f"  Erro: {erro['tipo_erro']} — "
                f"{erro['erro']}"
            )

    perfil_df = pd.DataFrame(perfis)
    erros_df = pd.DataFrame(erros)

    if not perfil_df.empty:
        perfil_df = perfil_df.sort_values(
            by=[
                "loja",
                "fonte",
                "ano",
                "arquivo",
                "posicao_coluna",
            ],
            na_position="last",
        ).reset_index(drop=True)

    return perfil_df, erros_df


def salvar_resultados(
    perfil: pd.DataFrame,
    erros: pd.DataFrame,
) -> None:
    """
    Salva os arquivos de profiling.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    perfil.to_csv(
        PROFILE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if erros.empty:
        if ERRORS_PATH.exists():
            ERRORS_PATH.unlink()

        return

    erros.to_csv(
        ERRORS_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def exibir_resumo(
    arquivos: list[Path],
    perfil: pd.DataFrame,
    erros: pd.DataFrame,
) -> None:
    """
    Exibe o resumo da execução.
    """
    arquivos_perfilados = (
        perfil["arquivo"].nunique()
        if not perfil.empty
        else 0
    )

    print("\nProfiling concluído.")
    print("Loja analisada: pilar")
    print(f"Arquivos encontrados: {len(arquivos)}")
    print(f"Arquivos perfilados: {arquivos_perfilados}")
    print(f"Colunas perfiladas: {len(perfil)}")
    print(f"Arquivos com erro: {len(erros)}")
    print(f"Perfil salvo em: {PROFILE_PATH}")

    if not erros.empty:
        print(f"Erros salvos em: {ERRORS_PATH}")


def main() -> None:
    """
    Executa o profiling da loja Pilar.
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Diretório da loja não encontrado: {INPUT_DIR}"
        )

    arquivos = localizar_arquivos_html(INPUT_DIR)

    if not arquivos:
        print(
            f"Nenhum arquivo HTML encontrado em: {INPUT_DIR}"
        )
        return

    print(f"{len(arquivos)} arquivo(s) encontrado(s).\n")

    perfil, erros = gerar_perfil(arquivos)

    salvar_resultados(
        perfil=perfil,
        erros=erros,
    )

    exibir_resumo(
        arquivos=arquivos,
        perfil=perfil,
        erros=erros,
    )


if __name__ == "__main__":
    main()