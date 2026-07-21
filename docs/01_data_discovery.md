## Escopo da versão 1

A primeira versão do Data Discovery está restrita à loja Pilar.

Os arquivos de outras lojas permanecem armazenados localmente, mas não
serão processados pelos scripts desta etapa.

## Descobertas estruturais iniciais

### Vendas por item

- O HTML contém duas tabelas.
- A tabela principal é a maior e possui sete colunas.
- Os cabeçalhos são reconhecidos corretamente pelo `pandas.read_html`.
- Uma linha representa um item vendido.
- Valores decimais são interpretados sem a vírgula decimal:
  - `1,00` é lido como `100`;
  - `85,00` é lido como `8500`.
- Códigos de produto são interpretados como `float`.

### Notas de venda

- A tabela possui dez colunas.
- Os cabeçalhos são reconhecidos corretamente.
- Valores monetários também são interpretados em centavos:
  - `124,00` é lido como `12400`.

### Estoque

- A tabela possui nove colunas.
- O campo `Grupo` será utilizado inicialmente como categoria do produto.
- Preço, quantidade e valor total apresentam o mesmo problema de escala
  decimal.
- Código do produto e código de barras devem ser tratados como texto.
- Deve ser investigado se zeros à esquerda fazem parte do código real.

### Compras

- A estrutura ainda precisa de inspeção específica.
- O inventário inicial identificou 25 colunas, possivelmente devido à
  estrutura interna do HTML.

  ## Resultados do inventário estrutural

- Escopo: loja Pilar.
- Total de arquivos processados: 27.
- Todas as fontes foram lidas sem erro.
- Fontes identificadas:
  - compras;
  - estoque;
  - notas de venda;
  - vendas por item.
- O relatório de compras possui 25 colunas físicas de layout e 9 colunas lógicas.
- Os relatórios de vendas por item possuem uma tabela externa de layout e uma tabela interna com os dados reais.
- Valores decimais brasileiros são interpretados pelo pandas sem a vírgula decimal.
- Códigos e códigos de barras precisam ser tratados como identificadores textuais.