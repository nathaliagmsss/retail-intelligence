"""
Recebe: 2018 2019 2020
Responde: Colunas iguais, Colunas diferentes, Mudanças, Tipos diferentes
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.discovery.analisar_html import OUTPUT_DIR


PROFILE_PATH = OUTPUT_DIR / "perfil_colunas.csv"

SCHEMA_COMPARISON_PATH = (
    OUTPUT_DIR / "comparacao_esquemas.csv"
)

SCHEMA_SUMMARY_PATH = (
    OUTPUT_DIR / "resumo_esquemas.csv"
)


def validar_colunas_perfil(perfil: pd.DataFrame) -> None:
    """
    Verifica se o arquivo de profiling possui as colunas necessárias.
    """
    colunas_obrigatorias = {
        "arquivo",
        "fonte",
        "ano",
        "coluna",
        "posicao_coluna",
        "tipo_pandas",
        "tipo_semantico_sugerido",
    }

    ausentes = colunas_obrigatorias.difference(perfil.columns)

    if ausentes:
        nomes = ", ".join(sorted(ausentes))

        raise ValueError(
            "O arquivo de perfil não contém todas as colunas "
            f"necessárias. Ausentes: {nomes}"
        )


def carregar_perfil(caminho: Path) -> pd.DataFrame:
    """
    Carrega o perfil das colunas gerado pelo profiler.py.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            "Arquivo de profiling não encontrado: "
            f"{caminho}\n"
            "Execute primeiro:\n"
            "python -m src.discovery.profiler"
        )

    perfil = pd.read_csv(
        caminho,
        encoding="utf-8-sig",
    )

    validar_colunas_perfil(perfil)

    perfil["ano"] = pd.to_numeric(
        perfil["ano"],
        errors="coerce",
    ).astype("Int64")

    perfil["posicao_coluna"] = pd.to_numeric(
        perfil["posicao_coluna"],
        errors="coerce",
    ).astype("Int64")

    return perfil


def formatar_lista_texto(
    valores: pd.Series,
) -> str:
    """
    Consolida valores distintos em texto ordenado.
    """
    itens = {
        str(valor).strip()
        for valor in valores.dropna()
        if str(valor).strip()
    }

    return " | ".join(sorted(itens))


def formatar_lista_numerica(
    valores: pd.Series,
) -> str:
    """
    Consolida números distintos em texto ordenado.
    """
    itens = sorted(
        {
            int(valor)
            for valor in valores.dropna()
        }
    )

    return " | ".join(str(valor) for valor in itens)


def obter_anos_por_fonte(
    perfil: pd.DataFrame,
) -> dict[str, set[int]]:
    """
    Retorna os anos disponíveis em cada fonte.
    """
    anos_por_fonte: dict[str, set[int]] = {}

    for fonte, grupo in perfil.groupby("fonte"):
        anos = {
            int(ano)
            for ano in grupo["ano"].dropna()
        }

        anos_por_fonte[str(fonte)] = anos

    return anos_por_fonte


def criar_comparacao_coluna(
    fonte: str,
    coluna: str,
    grupo: pd.DataFrame,
    anos_fonte: set[int],
) -> dict[str, object]:
    """
    Cria a comparação histórica de uma coluna dentro de uma fonte.
    """
    anos_presentes = {
        int(ano)
        for ano in grupo["ano"].dropna()
    }

    anos_ausentes = anos_fonte.difference(
        anos_presentes
    )

    tipos_pandas = {
        str(valor)
        for valor in grupo["tipo_pandas"].dropna()
    }

    tipos_semanticos = {
        str(valor)
        for valor in grupo[
            "tipo_semantico_sugerido"
        ].dropna()
    }

    posicoes = {
        int(valor)
        for valor in grupo["posicao_coluna"].dropna()
    }

    return {
        "fonte": fonte,
        "coluna": coluna,
        "anos_fonte": formatar_lista_numerica(
            pd.Series(sorted(anos_fonte))
        ),
        "anos_presentes": formatar_lista_numerica(
            pd.Series(sorted(anos_presentes))
        ),
        "anos_ausentes": formatar_lista_numerica(
            pd.Series(sorted(anos_ausentes))
        ),
        "presente_em_todos_anos": (
            anos_presentes == anos_fonte
        ),
        "quantidade_anos_fonte": len(anos_fonte),
        "quantidade_anos_presente": len(
            anos_presentes
        ),
        "tipos_pandas": " | ".join(
            sorted(tipos_pandas)
        ),
        "tipo_pandas_estavel": len(
            tipos_pandas
        ) <= 1,
        "tipos_semanticos": " | ".join(
            sorted(tipos_semanticos)
        ),
        "tipo_semantico_estavel": len(
            tipos_semanticos
        ) <= 1,
        "posicoes": " | ".join(
            str(valor)
            for valor in sorted(posicoes)
        ),
        "posicao_estavel": len(posicoes) <= 1,
        "arquivos_analisados": int(
            grupo["arquivo"].nunique()
        ),
    }


def gerar_comparacao_esquemas(
    perfil: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera uma linha por fonte e coluna.
    """
    anos_por_fonte = obter_anos_por_fonte(perfil)
    registros: list[dict[str, object]] = []

    agrupado = perfil.groupby(
        ["fonte", "coluna"],
        dropna=False,
    )

    for (fonte, coluna), grupo in agrupado:
        fonte_texto = str(fonte)
        coluna_texto = str(coluna)

        registro = criar_comparacao_coluna(
            fonte=fonte_texto,
            coluna=coluna_texto,
            grupo=grupo,
            anos_fonte=anos_por_fonte[
                fonte_texto
            ],
        )

        registros.append(registro)

    comparacao = pd.DataFrame(registros)

    if comparacao.empty:
        return comparacao

    return comparacao.sort_values(
        by=["fonte", "coluna"],
    ).reset_index(drop=True)


def contar_colunas_por_arquivo(
    perfil: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conta quantas colunas cada arquivo possui.
    """
    return (
        perfil
        .groupby(
            ["fonte", "arquivo", "ano"],
            dropna=False,
        )
        .agg(
            quantidade_colunas=(
                "coluna",
                "nunique",
            )
        )
        .reset_index()
    )


def obter_esquema_arquivo(
    grupo: pd.DataFrame,
) -> tuple[str, ...]:
    """
    Representa o esquema ordenado de um arquivo.

    A posição da coluna é considerada na comparação.
    """
    ordenado = grupo.sort_values(
        by="posicao_coluna"
    )

    return tuple(
        str(coluna)
        for coluna in ordenado["coluna"]
    )


def verificar_estabilidade_esquema(
    grupo_fonte: pd.DataFrame,
) -> bool:
    """
    Verifica se todos os arquivos de uma fonte possuem exatamente
    as mesmas colunas na mesma ordem.
    """
    esquemas: set[tuple[str, ...]] = set()

    for _, grupo_arquivo in grupo_fonte.groupby(
        ["arquivo", "ano"],
        dropna=False,
    ):
        esquemas.add(
            obter_esquema_arquivo(
                grupo_arquivo
            )
        )

    return len(esquemas) <= 1


def gerar_resumo_esquemas(
    perfil: pd.DataFrame,
    comparacao: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera um resumo por fonte.
    """
    contagem_colunas = contar_colunas_por_arquivo(
        perfil
    )

    registros: list[dict[str, object]] = []

    for fonte, grupo_fonte in perfil.groupby(
        "fonte"
    ):
        fonte_texto = str(fonte)

        arquivos_fonte = contagem_colunas[
            contagem_colunas["fonte"]
            == fonte
        ]

        comparacao_fonte = comparacao[
            comparacao["fonte"]
            == fonte_texto
        ]

        anos = sorted(
            {
                int(ano)
                for ano in grupo_fonte[
                    "ano"
                ].dropna()
            }
        )

        registros.append(
            {
                "fonte": fonte_texto,
                "arquivos": int(
                    grupo_fonte[
                        "arquivo"
                    ].nunique()
                ),
                "anos": " | ".join(
                    str(ano)
                    for ano in anos
                ),
                "ano_inicial": (
                    min(anos)
                    if anos
                    else None
                ),
                "ano_final": (
                    max(anos)
                    if anos
                    else None
                ),
                "quantidade_colunas_min": int(
                    arquivos_fonte[
                        "quantidade_colunas"
                    ].min()
                ),
                "quantidade_colunas_max": int(
                    arquivos_fonte[
                        "quantidade_colunas"
                    ].max()
                ),
                "colunas_distintas": int(
                    grupo_fonte[
                        "coluna"
                    ].nunique()
                ),
                "colunas_presentes_em_todos": int(
                    comparacao_fonte[
                        "presente_em_todos_anos"
                    ].sum()
                ),
                "colunas_com_ausencia": int(
                    (
                        ~comparacao_fonte[
                            "presente_em_todos_anos"
                        ]
                    ).sum()
                ),
                "colunas_com_tipo_pandas_instavel": int(
                    (
                        ~comparacao_fonte[
                            "tipo_pandas_estavel"
                        ]
                    ).sum()
                ),
                "colunas_com_tipo_semantico_instavel": int(
                    (
                        ~comparacao_fonte[
                            "tipo_semantico_estavel"
                        ]
                    ).sum()
                ),
                "colunas_com_posicao_instavel": int(
                    (
                        ~comparacao_fonte[
                            "posicao_estavel"
                        ]
                    ).sum()
                ),
                "esquema_estavel": (
                    verificar_estabilidade_esquema(
                        grupo_fonte
                    )
                ),
            }
        )

    resumo = pd.DataFrame(registros)

    if resumo.empty:
        return resumo

    return resumo.sort_values(
        by="fonte"
    ).reset_index(drop=True)


def salvar_resultados(
    comparacao: pd.DataFrame,
    resumo: pd.DataFrame,
) -> None:
    """
    Salva os CSVs da comparação de esquemas.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparacao.to_csv(
        SCHEMA_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    resumo.to_csv(
        SCHEMA_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def exibir_resumo_terminal(
    comparacao: pd.DataFrame,
    resumo: pd.DataFrame,
) -> None:
    """
    Exibe os principais resultados no terminal.
    """
    fontes = int(
        resumo["fonte"].nunique()
    ) if not resumo.empty else 0

    colunas_com_ausencia = int(
        (
            ~comparacao[
                "presente_em_todos_anos"
            ]
        ).sum()
    ) if not comparacao.empty else 0

    tipos_instaveis = int(
        (
            ~comparacao[
                "tipo_pandas_estavel"
            ]
        ).sum()
    ) if not comparacao.empty else 0

    posicoes_instaveis = int(
        (
            ~comparacao[
                "posicao_estavel"
            ]
        ).sum()
    ) if not comparacao.empty else 0

    print("\nComparação de esquemas concluída.")
    print(f"Fontes analisadas: {fontes}")
    print(
        "Combinações fonte-coluna: "
        f"{len(comparacao)}"
    )
    print(
        "Colunas ausentes em algum ano: "
        f"{colunas_com_ausencia}"
    )
    print(
        "Colunas com tipo pandas variável: "
        f"{tipos_instaveis}"
    )
    print(
        "Colunas com posição variável: "
        f"{posicoes_instaveis}"
    )
    print(
        "Comparação salva em: "
        f"{SCHEMA_COMPARISON_PATH}"
    )
    print(
        "Resumo salvo em: "
        f"{SCHEMA_SUMMARY_PATH}"
    )


def main() -> None:
    """
    Executa a comparação histórica dos esquemas.
    """
    perfil = carregar_perfil(PROFILE_PATH)

    if perfil.empty:
        print(
            "O arquivo de perfil está vazio. "
            "Nada para comparar."
        )
        return

    comparacao = gerar_comparacao_esquemas(
        perfil
    )

    resumo = gerar_resumo_esquemas(
        perfil=perfil,
        comparacao=comparacao,
    )

    salvar_resultados(
        comparacao=comparacao,
        resumo=resumo,
    )

    exibir_resumo_terminal(
        comparacao=comparacao,
        resumo=resumo,
    )


if __name__ == "__main__":
    main()