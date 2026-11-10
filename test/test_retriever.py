import time
inicio_total = time.perf_counter()
from llama_cloud import LlamaCloud
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate, SQLDatabase
from llama_index.core.postprocessor import SentenceTransformerRerank 
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import ObjectType
from dotenv import load_dotenv
import os
from pathlib import Path
tempo_total = time.perf_counter() - inicio_total




print(f"\nTempo total dos imports: {tempo_total:.4f} segundos")

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

openai_api_key = os.environ["OPENAI_API_KEY"]

groq_api_key = os.environ["GROQ_API_KEY"]

os.environ["HF_HUB_OFFLINE"] = "1"  # Desativando o download de modelos do Hugging Face Hub

#Instanciando o cliente da Llama Cloud com a chave da API
client = LlamaCloud(api_key=llama_cloud_key)

#Instanciando o llm
llm = OpenAI(model="gpt-5.6-terra",
           api_key=openai_api_key,
           request_timeout=300.0,
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
    simple_ratio_thresh=0.20
)

"""
    Utilizando o reranker como uma segunda camara de analise de similaridade entre os trechos para melhorar a qualidade das respostas
    O reranker é complementar ao parametro similarity_top_k do retriever, que é a primeira camada de analise de similaridade entre os trechos
    Juntos eles formam uma abordagem de duas camadas para melhorar a qualidade das respostas, onde o reranker é mais preciso e caro, enquanto 
    Similarity_top_k é mais rápido e barato, formando assim a Two-Stage Retrieval Approach, que é uma abordagem de duas camadas para melhorar a qualidade das respostas.
"""

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=10
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    retriever=automerging_retriever,
    llm=llm,
    node_postprocessors=[reranker],
    text_qa_template=prompt_template,
    streaming=False
)

while True:
    pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")

    if pergunta.lower().strip() == "sair":
        print("Encerrando o chatbot...")
        break

    if not pergunta.strip():
        print("Digite uma pergunta válida.")
        continue

    print("\n========== AUTO MERGING + RERANKER ==========")

    response = retriever_query_engine.query(pergunta)

    print("\n========== RESPOSTA COM RERANKER ==========")
    print(response)

    # ==========================================
    # DIAGNÓSTICO DOS NÓS RECUPERADOS
    # ==========================================

    print("\n========== DIAGNÓSTICO DO RAG ==========")

    source_nodes = response.source_nodes

    print(f"\nQuantidade de nós recuperados: {len(source_nodes)}")

    contexto_completo = []

    for i, node in enumerate(source_nodes, start=1):

        texto = node.node.get_content()

        print(f"\n{'=' * 60}")
        print(f"NÓ {i}")
        print(f"{'=' * 60}")

        print(f"Score: {node.score}")
        print(f"Tamanho: {len(texto)} caracteres")

        print("\nCONTEÚDO COMPLETO:")
        print(texto)

        contexto_completo.append(
            f"\n{'=' * 60}\n"
            f"NÓ {i}\n"
            f"Score: {node.score}\n"
            f"{'=' * 60}\n\n"
            f"{texto}"
        )

    # ==========================================
    # SALVAR CONTEXTO EM ARQUIVO
    # ==========================================

    with open(
        "contexto_recuperado.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(f"PERGUNTA: {pergunta}\n\n")

        f.write(
            f"QUANTIDADE DE NÓS: {len(source_nodes)}\n\n"
        )

        f.write("\n\n".join(contexto_completo))

    print("\n========== DIAGNÓSTICO FINALIZADO ==========")
    print("Contexto salvo em: contexto_recuperado.txt")