import re
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

def transform_md_csv(md_path, csv_path):
    md_path = Path(md_path)
    csv_path = Path(csv_path)
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if not md_path.exists():
        print(f"Ficheiro não encontrado: {md_path}")
        return
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all()
    dados_finais = []
    area_atual = ''
    #Variaeis de estado para tratar os diferentes rowspan
    prio_ret, nmo_ret, p3_ret, p2_ret, p1_ret = "", "", "", "", ""
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            # Ignora linhas de cabeçalho com <th>
            if row.find_all('th'):
                continue

            # Extrai o texto limpo de cada célula <td> (substituindo <br> por espaço)
            cols = []
            for td in row.find_all('td'):
                texto = td.get_text(separator=" ", strip=True)
                cols.append(texto)

            if not cols or all(c == "" for c in cols):
                continue

            # Caso 1: Linha completa com 7 colunas [Área, Curso, Prioritárias, NMO, Peso 3, Peso 2, Peso 1]
            if len(cols) == 7:
                area_atual = cols[0] if cols[0] != "" else area_atual
                curso, prio, nmo, p3, p2, p1 = cols[1], cols[2], cols[3], cols[4], cols[5], cols[6]

            # Caso 2: Linha de 6 colunas (A Área foi omitida por rowspan)
            elif len(cols) == 6:
                curso, prio, nmo, p3, p2, p1 = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]

            # Caso 3: Linha de 1 coluna (Subcursos com rowspan nas restantes colunas, ex: Música)
            elif len(cols) == 1:
                curso = cols[0]
                prio, nmo, p3, p2, p1 = prio_ret, nmo_ret, p3_ret, p2_ret, p1_ret

            else:
                continue

            # Atualiza os valores retidos para linhas subsequentes com rowspan
            prio_ret, nmo_ret, p3_ret, p2_ret, p1_ret = prio, nmo, p3, p2, p1

            dados_finais.append({
                'Area': area_atual,
                'Curso': curso,
                'Prioritarias': prio,
                'NMO': nmo,
                'Peso_3': p3,
                'Peso_2': p2,
                'Peso_1': p1
            })

    # Criação do DataFrame
    df = pd.DataFrame(dados_finais)

    # Limpeza e propagação da coluna Área (forward fill)
    df['Area'] = df['Area'].replace('', None).ffill()

    # Remoção da tabela de treineiros ou linhas inválidas, se necessário
    df = df[~df['Curso'].str.contains('Treineiros', na=False)]

    # Guardar em CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ Sucesso! {len(df)} cursos convertidos para CSV.")
    print(f"• Ficheiro guardado em: {csv_path.resolve()}")

if __name__ == "__main__":
    transform_md_csv(
        md_path="./data/anexo_3_extraido.md", 
        csv_path="./data/anexo_3_vagas.csv"
    )
    
