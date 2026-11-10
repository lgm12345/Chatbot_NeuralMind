import asyncio
import nest_asyncio
from llama_cloud import LlamaCloud
from llama_index.llms.openai_like import OpenAILike
from sqlalchemy import create_engine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate, SQLDatabase
from llama_index.core.postprocessor import SentenceTransformerRerank 
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, RetrieverTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from dotenv import load_dotenv
import os
from pathlib import Path
import pandas as pd
import streamlit as st
nest_asyncio.apply()

#Carregando o arquivo .env
load_dotenv()

#Criando um template para o prompt que sera enviado para o modelo de linguagem
prompt_template = PromptTemplate(
    "Você é um assistente especializado no Vestibular Unicamp 2027.\n"
    "As informações de contexto extraídas do edital estão abaixo:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dada a informação de contexto acima e não seu conhecimento prévio, "
    "responda à seguinte pergunta de forma clara, direta e em PORTUGUÊS.\n"
    "Se a informação não estiver expressamente descrita no contexto, "
    "informe educadamente que o edital fornecido não contém essa informação.\n\n"
    "Verifique rigorosamente os cabeçalhos das colunas (ex: 'Total Vagas Regulares', 'Total Vagas VU').\n"
    "Localize a linha EXATA do curso solicitado (atenção para não confundir 'Engenharia de Computação' com 'Ciência da Computação').\n"
    "REGRA DE FORMATAÇÃO: Sempre formate números de milhares utilizando o ponto como separador no padrão brasileiro"
    "REGRA DE FORMATAÇÃO: Não utilize nenhuma formatação Markdown (como asteriscos ** para negrito ou hashtags # para títulos). Retorne apenas texto puro sem formatação.\n\n"
    "Pergunta: {query_str}\n"
    "Resposta:"
)

#salvando as chaves
llama_cloud_key  = os.environ["LLAMA_CLOUD_API_KEY"]

groq_api_key = os.environ["GROQ_API_KEY"]

os.environ["HF_HUB_OFFLINE"] = "1"  # Desativando o download de modelos do Hugging Face Hub

#Instanciando o cliente da Llama Cloud com a chave da API
client = LlamaCloud(api_key=llama_cloud_key)

#Instanciando o llm
llm = OpenAILike(model="openai/gpt-oss-120b",
           api_key=groq_api_key ,
           api_base="https://api.groq.com/openai/v1",
           request_timeout=300.0,
           is_function_calling_model=True,
           is_chat_model=True,
           temperature=0.0)
Settings.llm = llm 
embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-m3')
Settings.embed_model = embed_model

#Carregando o banco de dados vetorial para ser usado nas consultas
persist_dir = Path("./storage")
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
vector_index = load_index_from_storage(storage_context=storage_context, index_cls=VectorStoreIndex)
base_retriever = vector_index.as_retriever(similarity_top_k=20)
# Configurando o AutoMergingRetriever para combinar os resultados do retriever com o reranker baseado na arquitetura em arvore dos documentos

automerging_retriever = AutoMergingRetriever(
    vector_retriever=base_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.25
)

"""
    Utilizando o reranker como uma segunda camara de analise de similaridade entre os trechos para melhorar a qualidade das respostas
    O reranker é complementar ao parametro similarity_top_k do retriever, que é a primeira camada de analise de similaridade entre os trechos
    Juntos eles formam uma abordagem de duas camadas para melhorar a qualidade das respostas, onde o reranker é mais preciso e caro, enquanto 
    Similarity_top_k é mais rápido e barato, formando assim a Two-Stage Retrieval Approach, que é uma abordagem de duas camadas para melhorar a qualidade das respostas.
"""
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=5
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    retriever=automerging_retriever,
    node_postprocessors=[reranker],
    llm=llm,
    text_qa_template=prompt_template,
    streaming=False
)

"""
    Utilizaremos tools para fazer uma busca hibrida no banco de dados vetorial ou no csv, com isso teremos mais precisao para cada caso.
"""

retriever_query_engine_tool =QueryEngineTool(
    query_engine=retriever_query_engine,
    metadata=ToolMetadata(
        name="regras_gerais_editais",
        description=(
            "Realiza busca semântica na base de textos do Edital e Manual do Candidato da Unicamp 2027."
            "Útil para consultar o CONTEÚDO PROGRAMÁTICO das provas (Matemática, Física, etc.), "
            "regras de isenção, valores de taxa, calendário, documentação e normas gerais."
            "REGRA DE INPUT: Passe a pergunta do usuário quase idêntica ao original"
            "PROIBIDO Usar conhecimento previo, se baseie apenas nas regras do edital"
    ),
    return_direct=True
)
)

df_anexo_3 = pd.read_csv("data/anexo_3_vagas.csv")

engine = create_engine("sqlite:///:memory:")
df_anexo_3.to_sql(name="tabela_vagas", con=engine, index=False)
sql_database = SQLDatabase(engine, include_tables=["tabela_vagas"])
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["tabela_vagas"],
)

sql_engine_tool = QueryEngineTool(
    query_engine=sql_query_engine,
    metadata=ToolMetadata(
        name="anexo_3_vagas.csv",
        description=(
            "Ferramenta para consultar dados estruturados da tabela anexo_3_vagas.csv  "
            "Contém a relação de Cursos, Áreas (Humanas/Artes, Biológicas, Exatas/Tecnológicas), "
            "Disciplinas Prioritárias, Nota Mínima de Opção (NMO) e os Pesos (Peso 1, Peso 2, Peso 3) "
            "de cada matéria (LPL, MAT, HIS, GEO, FIS, QUI, BIO, INTER, HE)."
        )

    )
)

agent = FunctionAgent(
    llm=llm,
    tools=[
        retriever_query_engine_tool,
        sql_engine_tool
    ],
    system_prompt = """
        Você é um assistente especializado no Vestibular Unicamp 2027.

        Utilize as ferramentas disponíveis para responder às perguntas.

        Para informações textuais do edital, utilize regras_gerais_editais.
        Para informações sobre vagas, cursos e pesos, utilize anexo_3_vagas.

        IMPORTANTE:
        - Preserve as informações fornecidas pelas ferramentas.
        - Não resuma respostas extensas, a menos que o usuário solicite.
        - Não omita listas, etapas, exceções ou detalhes importantes.
        - Não invente informações.
        - Responda em português brasileiro.
""",
    verbose=True,
    streaming=False,
    allow_parallel_tool_calls=False
)

async def loop_de_conversa(agent):
    # O contexto é inicializado ANTES do loop para manter a memória cumulativa da conversa
    
    print("Chatbot iniciado! Digite 'sair' para encerrar.")
    
    while True:

        pergunta = (await asyncio.to_thread(input, "\nQual a sua dúvida? ")).strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
            
        if not pergunta:
            continue
            
        print("\n[Agente a processar...]\n")
        
        # O agente processa a pergunta mantendo o contexto (ctx) das iterações anteriores
        resposta = await agent.run(user_msg=pergunta)
        
        print("\n--- RESPOSTA ---")
        print(resposta)
        print("-" * 50)



print("\n" + "="*50)
print("  CHATBOT VESTIBULAR UNICAMP 2027 (TESTE LOCAL)")
asyncio.run(loop_de_conversa(agent))
    

#Inicializacao da aplicação streamlit
st.title("Chatbot vestibular Unicamp 2025")
st.write("Faça perguntas sobre o vestibular da Unicamp")

#salvando a pergunta do usuario em uma string
pergunta = st.text_input("Qual a sua dúvida?")

if st.button("Enviar"):
    if pergunta:
        resposta = unicamp_engine.query(pergunta)
        st.write("Resposta")
        st.write(resposta.response)
    else:
        st.write("Por favor digite uma pergunta:")
