import pandas as pd
from llama_index.core.query_engine import NLSQLTableQueryEngine     
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine
import os
from llama_cloud import LlamaCloud
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

#salvando as chaves
llama_cloud_key  = os.environ["LLAMA_CLOUD_API_KEY"]

openai_api_key = os.environ["OPENAI_API_KEY"]


os.environ["HF_HUB_OFFLINE"] = "1"  # Desativando o download de modelos do Hugging Face Hub

#Instanciando o cliente da Llama Cloud com a chave da API
client = LlamaCloud(api_key=llama_cloud_key)

#Instanciando o llm
llm = OpenAI(model="gpt-5.6-terra",
           api_key=openai_api_key,
           request_timeout=300.0,
           temperature=0.0)
contexto_sql = """
A tabela tabela_vagas contém informações sobre os cursos
do Vestibular Unicamp 2027.

COLUNAS:
- Area: área do conhecimento.
- Curso: nome do curso.
- Prioritarias: disciplinas prioritárias.
- NMO: nota mínima de opção.
- Peso_3: disciplinas com peso 3.
- Peso_2: disciplinas com peso 2.
- Peso_1: disciplinas com peso 1.

DICIONÁRIO DE DISCIPLINAS:
- MAT = Matemática
- FIS = Física
- QUI = Química
- BIO = Biologia
- HE = História
- LPL = Língua Portuguesa e Literaturas

REGRAS IMPORTANTES:

1. As disciplinas são armazenadas utilizando SIGLAS,
   não seus nomes completos.

2. Quando o usuário mencionar uma disciplina pelo nome,
   converta o nome para a sigla correspondente antes
   de gerar a consulta SQL.

3. As colunas Prioritarias, Peso_3, Peso_2 e Peso_1
   armazenam múltiplas siglas separadas por ponto e vírgula.

4. Para pesquisar uma disciplina, utilize uma comparação
   que respeite os delimitadores.

EXEMPLO:

Pergunta:
Quais cursos possuem Matemática como disciplina prioritária?

SQL:
SELECT Curso
FROM tabela_vagas
WHERE ';' || REPLACE(Prioritarias, ' ', '') || ';'
LIKE '%;MAT;%';

5. O caractere '-' representa ausência de informação.
"""

df_anexo_3 = pd.read_csv("data/anexo_3_vagas.csv")

engine = create_engine("sqlite:///:memory:")
df_anexo_3.to_sql(name="tabela_vagas", con=engine, index=False)
sql_database = SQLDatabase(engine, include_tables=["tabela_vagas"])
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["tabela_vagas"],
    llm=llm,
    context_query_kwargs={
        "tabela_vagas": contexto_sql
    }
)

while True:
    user_input = input("Digite sua pergunta (ou 'sair' para encerrar): ")
    if user_input.lower() == 'sair':
        break
    response = sql_query_engine.query(user_input)
    print("Resposta:", response)



