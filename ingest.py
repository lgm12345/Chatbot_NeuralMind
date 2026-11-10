import os
import re
from pathlib import Path
from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

def ingest_pdf():
    """
        Funcão que realiza a ingestão do arquivo PDF, vetoriza o conteúdo, cria os vetor de indices e salva em um banco de dados vetorial local.
        Criamos uma referencia para o arquivo PDF e setamos o purpose como 'parse' para que seja feito o parse em seguida.
        O parametro tier do metodo parse indica metodo que sera utilizado para realzar o parse, sendo 'agentic' o metodo mais avançado e preciso(Mais caro tambem)
    """
    pdf_path = Path("./data/unicamp_2027_edital.pdf")
    persist_dir = Path("./storage")
    llama_cloud_key = os.environ["LLAMA_CLOUD_API_KEY"]
    client = LlamaCloud(api_key=llama_cloud_key)
    file_ref = client.files.create(file=pdf_path,
                                   purpose='parse')
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1536, 384],
                                                            chunk_overlap=60)
    result = client.parsing.parse(
        file_id=file_ref.id,
        version='2026-10-15',
        tier='agentic',
        expand=["markdown"]
    )

    documents = []

    for page in result.markdown.pages:
        if not page.success:
            continue
        page_text = (page.markdown or "").strip()
        if not page_text:
            continue
        document = Document(
            text=page_text,
            metadata={
                "file_name": pdf_path.name,
                "page_number": page.page_number
            }
        )
        documents.append(document)

    #String de padrao para percorrer a string com a biblioteca padrao re-> divide em grupos de acordo com o conteudo dos parenteses
    # Gera a lista completa com todos os nós (pais e filhos)
    all_nodes = node_parser.get_nodes_from_documents(documents)
    # Separa apenas os nós folha para vetorização no VectorStoreIndex
    leaf_nodes = get_leaf_nodes(all_nodes)
    #Configurar o modelo de Embeddings
    embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-m3')
    Settings.embed_model = embed_model
    #Configurar o Divisor de Texto (SentenceSplitter)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(all_nodes)
    index = VectorStoreIndex(storage_context=storage_context, embed_model=embed_model, nodes=leaf_nodes)
    #Criando a persistindo o bando de dados vetorial localmente para que seja possivel carregar em outra sessão sem precisar realizar a ingestão novamente
    storage_context.persist(persist_dir)
    print("Ingestão concluida com sucesso!)")

if __name__ == "__main__":
    ingest_pdf()



