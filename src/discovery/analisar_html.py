from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOJA_ALVO = "pilar"
INPUT_DIR = RAW_DATA_DIR / LOJA_ALVO

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "discovery"
INVENTORY_PATH = OUTPUT_DIR / "inventario_arquivos.csv"


ESQUEMAS_ESPERADOS: dict[str, list[str]] = {
    "compras": [
        "NF Número",
        "Série",
        "Mod.",
        "Emissão",
        "Entrada",
        "Ident.",
        "Descrição",
        "Quant.",
        "Custo Compra",
    ],
    "estoque": [
        "Código",
        "Código Barras",
        "Descrição",
        "Grupo",
        "Preço em R$",
        "Quantidade",
        "Últ.Compra",
        "Últ.Venda",
        "Qtd X preço",
    ],
    "notas_venda": [
        "Núm./Série",
        "Nat. operação",
        "Vendedor",
        "Cliente",
        "Emissão",
        "Mercadorias",
        "Desc./Acr.Prod.",
        "TOTAL",
        "Hora saída",
        "Data saída",
    ],
    "vendas_itens": [
        "Nota",
        "Data",
        "Código",
        "Descrição do item",
        "Quantidade",
        "Vendido por",
        "Custo compra",
    ],
}


def localizar_arquivos_html(diretorio: Path) -> list[Path]:
    """Localiza recursivamente arquivos .html e .htm."""
    extensoes_validas = {".html", ".htm"}

    return sorted(
        caminho
        for caminho in diretorio.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in extensoes_validas
    )


def extrair_ano(nome_arquivo: str) -> int | None:
    """Extrai um ano entre 2000 e 2099 do nome do arquivo."""
    resultado = re.search(r"(?<!\d)(20\d{2})(?!\d)", nome_arquivo)

    if resultado is None:
        return None

    return int(resultado.group(1))


def extrair_metadados_arquivo(caminho: Path) -> dict[str, Any]:
    """
    Extrai metadados da estrutura:

    data/raw/<loja>/<fonte>/<arquivo>
    """
    caminho_relativo = caminho.relative_to(RAW_DATA_DIR)
    partes = caminho_relativo.parts

    return {
        "arquivo": caminho.name,
        "caminho_relativo": caminho_relativo.as_posix(),
        "loja": partes[0] if len(partes) >= 1 else None,
        "fonte": partes[1] if len(partes) >= 2 else None,
        "ano": extrair_ano(caminho.name),
        "extensao": caminho.suffix.lower(),
        "tamanho_bytes": caminho.stat().st_size,
    }


def normalizar_texto(valor: Any) -> str:
    """Normaliza um texto para comparação de cabeçalhos."""
    texto = str(valor).strip().casefold()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def calcular_tamanho_tabela(tabela: pd.DataFrame) -> int:
    """Retorna a quantidade física de células do DataFrame."""
    linhas, colunas = tabela.shape
    return linhas * max(colunas, 1)


def formatar_colunas(colunas: list[Any]) -> str:
    """Converte nomes de colunas em uma string legível."""
    return " | ".join(str(coluna) for coluna in colunas)


def formatar_primeira_linha(tabela: pd.DataFrame) -> str | None:
    """Formata a primeira linha não totalmente vazia."""
    if tabela.empty:
        return None

    linhas_validas = tabela.dropna(how="all")

    if linhas_validas.empty:
        return None

    valores = [
        "" if pd.isna(valor) else str(valor).strip()
        for valor in linhas_validas.iloc[0].tolist()
    ]

    return " | ".join(valores)


def ler_sopa_html(caminho: Path) -> BeautifulSoup:
    """Lê o HTML preservando a estrutura original."""
    conteudo = caminho.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return BeautifulSoup(conteudo, "html.parser")


def extrair_textos_celulas(linha: Any) -> list[str]:
    """Extrai os textos das células diretamente filhas de uma linha."""
    return [
        celula.get_text(" ", strip=True)
        for celula in linha.find_all(
            ["td", "th"],
            recursive=False,
        )
    ]


def pontuar_esquema(
    tabela: pd.DataFrame,
    colunas_esperadas: list[str],
) -> int:
    """
    Mede quantos cabeçalhos esperados aparecem nas colunas da tabela.

    Quanto maior a pontuação, maior a chance de ser a tabela real.
    """
    colunas_encontradas = {
        normalizar_texto(coluna)
        for coluna in tabela.columns
    }

    esperadas = {
        normalizar_texto(coluna)
        for coluna in colunas_esperadas
    }

    return len(colunas_encontradas.intersection(esperadas))


def selecionar_tabela_principal(
    tabelas: list[pd.DataFrame],
    fonte: str,
) -> tuple[int, pd.DataFrame, int]:
    """
    Seleciona a tabela principal pelo esquema esperado.

    Em caso de empate, escolhe a maior tabela.
    """
    esquema = ESQUEMAS_ESPERADOS.get(fonte, [])

    candidatas = []

    for indice, tabela in enumerate(tabelas):
        pontuacao = pontuar_esquema(tabela, esquema)

        candidatas.append(
            (
                pontuacao,
                calcular_tamanho_tabela(tabela),
                indice,
                tabela,
            )
        )

    pontuacao, _, indice, tabela = max(
        candidatas,
        key=lambda item: (item[0], item[1]),
    )

    return indice, tabela, pontuacao


def encontrar_cabecalho_compras(
    sopa: BeautifulSoup,
) -> tuple[int | None, list[str] | None]:
    """Localiza o cabeçalho lógico do relatório de compras."""
    esperado = ESQUEMAS_ESPERADOS["compras"]
    esperado_normalizado = [
        normalizar_texto(valor)
        for valor in esperado
    ]

    for indice, linha in enumerate(sopa.find_all("tr")):
        textos = extrair_textos_celulas(linha)
        textos_normalizados = [
            normalizar_texto(valor)
            for valor in textos
        ]

        if textos_normalizados == esperado_normalizado:
            return indice, textos

    return None, None


def linha_valida_compras(textos: list[str]) -> bool:
    """Verifica se uma linha possui a estrutura de um item de compra."""
    if len(textos) != 9:
        return False

    padrao_data = re.compile(r"^\d{2}/\d{2}/\d{4}$")

    return bool(
        padrao_data.fullmatch(textos[3])
        and padrao_data.fullmatch(textos[4])
    )


def contar_linhas_dados_compras(
    sopa: BeautifulSoup,
    indice_cabecalho: int,
) -> int:
    """Conta registros lógicos do relatório de compras."""
    linhas = sopa.find_all("tr")

    return sum(
        1
        for linha in linhas[indice_cabecalho + 1 :]
        if linha_valida_compras(
            extrair_textos_celulas(linha)
        )
    )


def extrair_primeira_linha_compras(
    sopa: BeautifulSoup,
    indice_cabecalho: int,
) -> str | None:
    """Extrai o primeiro registro lógico de compras."""
    linhas = sopa.find_all("tr")

    for linha in linhas[indice_cabecalho + 1 :]:
        textos = extrair_textos_celulas(linha)

        if linha_valida_compras(textos):
            return " | ".join(textos)

    return None


def analisar_compras(
    caminho: Path,
    registro: dict[str, Any],
) -> dict[str, Any]:
    """
    Analisa compras pela estrutura lógica do HTML.

    O FastReport utiliza 25 colunas físicas de layout, mas existem
    nove campos lógicos.
    """
    sopa = ler_sopa_html(caminho)
    tabelas_html = sopa.find_all("table")

    registro["quantidade_tabelas"] = len(tabelas_html)
    registro["tabela_principal"] = 0 if tabelas_html else None

    indice_cabecalho, cabecalhos = encontrar_cabecalho_compras(sopa)

    if indice_cabecalho is None or cabecalhos is None:
        registro["status"] = "cabecalho_nao_encontrado"
        registro["erro"] = (
            "Cabeçalho lógico do relatório de compras não encontrado."
        )
        return registro

    registro["linhas"] = contar_linhas_dados_compras(
        sopa,
        indice_cabecalho,
    )
    registro["colunas_fisicas"] = 25
    registro["colunas_logicas"] = len(cabecalhos)
    registro["nomes_colunas"] = formatar_colunas(cabecalhos)
    registro["primeira_linha"] = extrair_primeira_linha_compras(
        sopa,
        indice_cabecalho,
    )
    registro["pontuacao_esquema"] = len(cabecalhos)

    return registro


def analisar_tabela_padrao(
    caminho: Path,
    registro: dict[str, Any],
) -> dict[str, Any]:
    """Analisa fontes reconhecidas diretamente pelo pandas."""
    tabelas = pd.read_html(caminho)

    registro["quantidade_tabelas"] = len(tabelas)

    if not tabelas:
        registro["status"] = "sem_tabelas"
        return registro

    indice, tabela, pontuacao = selecionar_tabela_principal(
        tabelas=tabelas,
        fonte=str(registro["fonte"]),
    )

    registro["tabela_principal"] = indice
    registro["linhas"] = tabela.shape[0]
    registro["colunas_fisicas"] = tabela.shape[1]
    registro["colunas_logicas"] = tabela.shape[1]
    registro["nomes_colunas"] = formatar_colunas(
        list(tabela.columns)
    )
    registro["primeira_linha"] = formatar_primeira_linha(tabela)
    registro["pontuacao_esquema"] = pontuacao

    esquema = ESQUEMAS_ESPERADOS.get(
        str(registro["fonte"]),
        [],
    )

    if esquema and pontuacao == 0:
        registro["status"] = "esquema_nao_reconhecido"
        registro["erro"] = (
            "Nenhum cabeçalho esperado foi encontrado."
        )

    return registro


def analisar_arquivo_html(caminho: Path) -> dict[str, Any]:
    """Analisa estruturalmente um arquivo HTML."""
    registro = extrair_metadados_arquivo(caminho)

    registro.update(
        {
            "status": "sucesso",
            "quantidade_tabelas": 0,
            "tabela_principal": None,
            "linhas": None,
            "colunas_fisicas": None,
            "colunas_logicas": None,
            "pontuacao_esquema": None,
            "nomes_colunas": None,
            "primeira_linha": None,
            "erro": None,
        }
    )

    try:
        if registro["fonte"] == "compras":
            return analisar_compras(caminho, registro)

        return analisar_tabela_padrao(caminho, registro)

    except Exception as exc:
        registro["status"] = "erro"
        registro["erro"] = f"{type(exc).__name__}: {exc}"
        return registro


def gerar_inventario(arquivos: list[Path]) -> pd.DataFrame:
    """Analisa todos os arquivos e consolida o inventário."""
    registros: list[dict[str, Any]] = []
    total = len(arquivos)

    for indice, caminho in enumerate(arquivos, start=1):
        caminho_relativo = caminho.relative_to(PROJECT_ROOT)

        print(
            f"[{indice}/{total}] "
            f"Analisando: {caminho_relativo.as_posix()}"
        )

        registro = analisar_arquivo_html(caminho)
        registros.append(registro)

        if registro["status"] != "sucesso":
            print(
                f"  Status: {registro['status']} — "
                f"{registro['erro']}"
            )

    inventario = pd.DataFrame(registros)

    if not inventario.empty:
        inventario = inventario.sort_values(
            by=["loja", "fonte", "ano", "arquivo"],
            na_position="last",
        ).reset_index(drop=True)

    return inventario


def salvar_inventario(
    inventario: pd.DataFrame,
    destino: Path,
) -> None:
    """Salva o inventário em CSV."""
    destino.parent.mkdir(parents=True, exist_ok=True)

    inventario.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig",
    )


def exibir_resumo(inventario: pd.DataFrame) -> None:
    """Exibe o resumo final no terminal."""
    sucessos = int((inventario["status"] == "sucesso").sum())
    outros = len(inventario) - sucessos

    print("\nAnálise concluída.")
    print(f"Loja analisada: {LOJA_ALVO}")
    print(f"Arquivos processados: {len(inventario)}")
    print(f"Sucessos: {sucessos}")
    print(f"Outros status/erros: {outros}")
    print(f"Inventário salvo em: {INVENTORY_PATH}")


def main() -> None:
    """Executa o inventário estrutural da loja Pilar."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Diretório da loja não encontrado: {INPUT_DIR}"
        )

    arquivos = localizar_arquivos_html(INPUT_DIR)

    if not arquivos:
        print(f"Nenhum arquivo HTML encontrado em: {INPUT_DIR}")
        return

    print(f"Loja alvo: {LOJA_ALVO}")
    print(f"{len(arquivos)} arquivo(s) HTML encontrado(s).\n")

    inventario = gerar_inventario(arquivos)
    salvar_inventario(inventario, INVENTORY_PATH)
    exibir_resumo(inventario)


if __name__ == "__main__":
    main()