import asyncio
import os
import time
from pathlib import Path
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    SQLDatabase,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    NLSQLTableQueryEngine,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from contextvars import ContextVar
import streamlit as st

nest_asyncio.apply()

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

CSV_PATH = Path("data/anexo_3_vagas.csv")
STORAGE_PATH = Path("./storage")

# True: mantém a memória entre perguntas.
# False: testa cada pergunta independentemente.
MANTER_CONTEXTO = True

# True: imprime detalhes das ferramentas.
DEBUG = True

llm = OpenAI(
    model="gpt-5.6-terra",
    api_key=OPENAI_API_KEY,
    request_timeout=120.0,
    max_retries=1,
    temperature=0.0,
    reasoning_effort='none'
)

Settings.llm = llm

os.environ["HF_HUB_OFFLINE"] = "1"

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

Settings.embed_model = embed_model

pergunta_original_ctx = ContextVar(
    "pergunta_original",
    default=None
)

# ============================================================
# 4. PROMPT DO RAG
# ============================================================

prompt_template = PromptTemplate(
    """
Você é um assistente especializado no Vestibular Unicamp 2027.

As informações extraídas do edital estão abaixo:

---------------------
{context_str}
---------------------

Responda à pergunta utilizando exclusivamente o contexto fornecido.

REGRAS:
1. Responda em português brasileiro.
2. Não utilize conhecimento prévio para completar informações.
3. Não invente datas, números, regras ou conteúdos.
4. Preserve listas, etapas e exceções relevantes.
5. Se a pergunta solicitar todos os conteúdos de uma disciplina,
   apresente todos os conteúdos disponíveis no contexto.
6. Se a informação não estiver no contexto, informe a limitação.
7. Utilize o ponto como separador de milhares.
8. Retorne texto puro, sem formatação Markdown.

Pergunta: {query_str}

Resposta:
"""
)


# ============================================================
# 5. CARREGAMENTO DO ÍNDICE VETORIAL
# ============================================================

if not STORAGE_PATH.exists():
    raise FileNotFoundError(
        f"Diretório do índice não encontrado: {STORAGE_PATH}"
    )

print("\n[RAG] Carregando índice vetorial...", flush=True)

storage_context = StorageContext.from_defaults(
    persist_dir=str(STORAGE_PATH)
)

vector_index = load_index_from_storage(
    storage_context=storage_context,
    index_cls=VectorStoreIndex,
)

base_retriever = vector_index.as_retriever(
    similarity_top_k=20
)

automerging_retriever = AutoMergingRetriever(
    vector_retriever=base_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.25,
)

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=10,
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    retriever=automerging_retriever,
    node_postprocessors=[reranker],
    llm=llm,
    text_qa_template=prompt_template,
    response_mode="compact",
    streaming=False,
)

print("[RAG] Inicializado com sucesso.", flush=True)


# ============================================================
# 6. CARREGAMENTO DO CSV
# ============================================================

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={
        "check_same_thread": False
    },
    poolclass=StaticPool)
df_anexo_3 = pd.read_csv("data/anexo_3_vagas.csv")
df_anexo_3.to_sql(
    name="tabela_vagas",
    con=engine,
    index=False,
    if_exists="replace",
)

# O LLM não deve receber acesso de escrita.
# O modo query_only é ativado após a importação do CSV.
@event.listens_for(engine, "connect")
def configurar_sqlite(dbapi_connection, connection_record):
    dbapi_connection.execute("PRAGMA query_only = ON")


# Aplica também à conexão já criada pelo to_sql().
with engine.connect() as conn:
    conn.exec_driver_sql("PRAGMA query_only = ON")


sql_database = SQLDatabase(
    engine,
    include_tables=["tabela_vagas"],
)


# ============================================================
# 8. PROMPT SQL
# ============================================================

text_to_sql_prompt = PromptTemplate(
    """
Você é um especialista em SQLite e nos dados estruturados
do Vestibular Unicamp 2027.

Converta a pergunta do usuário em uma consulta SQL válida.

REGRAS:
- Gere somente SELECT.
- Utilize apenas tabelas e colunas existentes.
- Não invente valores.
- Utilize a sintaxe SQLite.
- Para contagens, utilize COUNT.
- Para consultas de disciplinas, utilize suas siglas.

TABELA:
tabela_vagas

COLUNAS:
- Area: área do conhecimento.
- Curso: nome do curso.
- Prioritarias: disciplinas prioritárias.
- NMO: nota mínima de opção.
- Peso_3: disciplinas com peso 3.
- Peso_2: disciplinas com peso 2.
- Peso_1: disciplinas com peso 1.

DICIONÁRIO DE DISCIPLINAS:

MAT = Matemática
FIS = Física
QUI = Química
BIO = Biologia
HE = História
LPL = Língua Portuguesa e Literaturas

IMPORTANTE:
Confirme e complete esse dicionário com a legenda oficial
do seu CSV antes de disponibilizar o sistema publicamente.

REGRAS PARA DISCIPLINAS:

1. As disciplinas estão armazenadas como siglas.

2. As colunas Prioritarias, Peso_3, Peso_2 e Peso_1
   podem conter múltiplas siglas separadas por ponto e vírgula.

3. Ao buscar uma disciplina, respeite os delimitadores.

4. Para Matemática como prioritária, utilize:

SELECT Curso
FROM tabela_vagas
WHERE ';' || REPLACE(Prioritarias, ' ', '') || ';'
LIKE '%;MAT;%';

5. Para Matemática E Física como prioritárias,
   utilize duas condições combinadas com AND.

6. O caractere '-' representa ausência de informação.

7. Valores NULL de NMO não devem satisfazer filtros numéricos.

EXEMPLO:

Pergunta:
Quais cursos possuem Matemática e Física como prioritárias?

SQL:
SELECT Curso
FROM tabela_vagas
WHERE ';' || REPLACE(Prioritarias, ' ', '') || ';'
LIKE '%;MAT;%'
AND ';' || REPLACE(Prioritarias, ' ', '') || ';'
LIKE '%;FIS;%';

---------------------

Dialeto: {dialect}

Esquema disponível:
{schema}

Pergunta:
{query_str}

SQLQuery:
"""
)


# ============================================================
# 9. SQL QUERY ENGINE
# ============================================================

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

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["tabela_vagas"],
    llm=llm,
    context_query_kwargs={
        "tabela_vagas": contexto_sql
    }
)

print("[SQL] Inicializada com sucesso.", flush=True)


# ============================================================
# 10. FUNÇÕES DAS TOOLS
# ============================================================

def consultar_edital(pergunta: str) -> str:
    """
    Consulta o edital utilizando exatamente a pergunta original
    digitada pelo usuário.
    """

    inicio = time.perf_counter()

    # Recupera a pergunta original armazenada no contexto.
    pergunta_original = pergunta_original_ctx.get()

    if pergunta_original is None:
        raise RuntimeError(
            "Pergunta original não encontrada no contexto."
        )

    print("\n" + "=" * 60)
    print("[TOOL RAG] DIAGNÓSTICO")
    print("=" * 60)

    print(f"Pergunta enviada pelo agente: {pergunta}")
    print(f"Pergunta ORIGINAL: {pergunta_original}")

    # IMPORTANTE:
    # O RAG recebe a pergunta original, não a reformulada.
    response = retriever_query_engine.query(
        pergunta_original
    )

    tempo = time.perf_counter() - inicio

    print(f"\n[TOOL RAG] Tempo: {tempo:.2f}s")

    print(
        f"[TOOL RAG] Nós utilizados: {len(response.source_nodes)}"
    )

    return str(response)


def consultar_dados_cursos(pergunta: str) -> str:
    """
    Consulta dados estruturados dos cursos da Unicamp 2027,
    utilizando exatamente a pergunta original do usuário.
    """

    inicio = time.perf_counter()

    pergunta_original = pergunta_original_ctx.get()

    if pergunta_original is None:
        raise RuntimeError(
            "Pergunta original não encontrada no contexto."
        )

    print("\n" + "=" * 60)
    print("[TOOL SQL] DIAGNÓSTICO")
    print("=" * 60)

    print(f"Pergunta do agente: {pergunta}")
    print(f"Pergunta original: {pergunta_original}")

    try:
        # Utiliza a pergunta original, ignorando a reformulação.
        response = sql_query_engine.query(pergunta_original)

        tempo = time.perf_counter() - inicio

        print(f"\n[TOOL SQL] Concluída em {tempo:.2f}s")

        if DEBUG:
            metadata = getattr(response, "metadata", {}) or {}

            print(
                "[TOOL SQL] SQL gerado:",
                metadata.get("sql_query")
            )

            print(
                "[TOOL SQL] Resultado bruto:",
                metadata.get("result")
            )

        return str(response)

    except Exception as e:
        tempo = time.perf_counter() - inicio

        print(
            f"[TOOL SQL] ERRO: {type(e).__name__}: {e}"
        )

        print(f"[TOOL SQL] Tempo até o erro: {tempo:.2f}s")

        raise


# ============================================================
# 11. REGISTRO DAS TOOLS
# ============================================================

rag_tool = FunctionTool.from_defaults(
    fn=consultar_edital,
    name="consultar_edital",
    description=(
        "Consulta o texto do edital Unicamp 2027. "
        "Use para datas, inscrições, calendário, documentação, "
        "conteúdos programáticos de Física, Química, Matemática "
        "e outras disciplinas, regras e critérios de classificação. "
        "Não use para consultar pesos, NMO ou disciplinas "
        "prioritárias de cursos."
    ),
    return_direct=True
)

sql_tool = FunctionTool.from_defaults(
    fn=consultar_dados_cursos,
    name="consultar_dados_cursos",
    description=(
        "Consulta o banco SQLite do Anexo III da Unicamp 2027. "
        "Use para cursos, áreas, notas mínimas de opção (NMO), "
        "disciplinas prioritárias e pesos 1, 2 e 3. "
        "Permite filtros, contagens e comparações entre cursos. "
        "Não use para consultar conteúdos programáticos ou "
        "regras gerais do edital."
    ),
)


# ============================================================
# 12. FUNCTION AGENT
# ============================================================

agent = FunctionAgent(
    llm=llm,
    tools=[
        rag_tool,
        sql_tool,
    ],
    system_prompt="""
        Você é um assistente especializado no Vestibular Unicamp 2027.

        Sua função é responder às perguntas utilizando as ferramentas
        disponíveis.

        FERRAMENTAS:

        1. consultar_edital:
        - Datas e calendário.
        - Inscrições.
        - Regras gerais.
        - Conteúdos programáticos.
        - Funcionamento das provas.
        - Critérios de classificação.

        2. consultar_dados_cursos:
        - Cursos e áreas.
        - Disciplinas prioritárias.
        - NMO.
        - Pesos das disciplinas.
        - Comparações entre cursos.

        REGRAS DE SELEÇÃO:

        - Se a pergunta tratar de conteúdos programáticos,
        utilize consultar_edital.

        - Se a pergunta tratar de pesos, NMO ou disciplinas
        prioritárias, utilize consultar_dados_cursos.

        - Se a pergunta exigir informações textuais e estruturadas,
        utilize AMBAS as ferramentas antes de responder.

        REGRAS DE RESPOSTA:

        - Consulte as ferramentas antes de responder perguntas factuais.
        - Não invente informações.
        - Não utilize conhecimento prévio para completar lacunas.
        - Preserve listas completas quando solicitadas.
        - Não omita detalhes importantes.
        - Não confunda disciplinas prioritárias com conteúdos
        programáticos.
        - Se uma ferramenta não fornecer a informação necessária,
        informe a limitação.
        - Responda sempre em português brasileiro.
        - Utilize texto puro, sem formatação Markdown.
        """,
    streaming=False,
    allow_parallel_tool_calls=False,
)

# ============================================================
# 13. INTERFACE STREAMLIT
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Chatbot Unicamp 2027",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Chatbot Vestibular Unicamp 2027")

st.write(
    "Faça perguntas sobre o edital, cursos, vagas e "
    "conteúdos programáticos do Vestibular Unicamp 2027."
)

# ============================================================
# 14. GERENCIAMENTO DA SESSÃO
# ============================================================

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "ctx" not in st.session_state:
    st.session_state.ctx = Context(agent)


# ============================================================
# 15. FUNÇÃO DE EXECUÇÃO DO AGENTE
# ============================================================

async def executar_agente(pergunta, ctx):

    # Preserva a pergunta original do usuário
    token = pergunta_original_ctx.set(pergunta)

    try:
        resposta = await agent.run(
            user_msg=pergunta,
            ctx=ctx
        )

        return str(resposta)

    finally:
        pergunta_original_ctx.reset(token)


# ============================================================
# 16. HISTÓRICO DE MENSAGENS
# ============================================================

for mensagem in st.session_state.mensagens:

    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])


# ============================================================
# 17. CHATBOT
# ============================================================

pergunta = st.chat_input("Qual é a sua dúvida?")

if pergunta and pergunta.strip():

    pergunta = pergunta.strip()

    # Armazena a mensagem do usuário
    st.session_state.mensagens.append({
        "role": "user",
        "content": pergunta
    })

    with st.chat_message("user"):
        st.markdown(pergunta)

    # Executa o agente
    with st.chat_message("assistant"):

        with st.spinner("Consultando informações do vestibular..."):

            try:

                resposta = asyncio.run(
                    executar_agente(
                        pergunta,
                        st.session_state.ctx
                    )
                )

                st.markdown(resposta)

                # Armazena a resposta
                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": resposta
                })

            except Exception as e:

                st.error(
                    "Ocorreu um erro ao processar sua pergunta."
                )

                st.exception(e)