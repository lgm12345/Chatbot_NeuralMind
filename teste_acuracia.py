from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
import os

# Carregando o arquivo .env
load_dotenv()

# Salvando as chaves
openai_key = os.environ["OPENAI_API_KEY"]
llama_cloud_key  = os.environ["LLAMA_CLOUD_API_KEY"]

# Instanciando o llm da OpenAI
llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_key, timeout=30.0)

# Instanciando o objeto LlamaParse
parser = LlamaParse(result_type="markdown", api_key=llama_cloud_key, language="pt")
file_extractor = {".txt":parser}
Settings.embed_model = OpenAIEmbedding()
embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=50)

# Carregando os dados do diretório data
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)
unicamp_engine = vector_index.as_query_engine(llm=llm)

# Função para calcular a acurácia
def calcular_acuracia(file_path):
    total_perguntas = 0
    respostas_corretas = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for linha in f:
            if '|' in linha:
                pergunta, resposta_esperada = linha.strip().split('|')
                resposta = unicamp_engine.query(pergunta).response
                
                # Comparar as palavras da resposta esperada e da resposta fornecida
                palavras_esperadas = set(resposta_esperada.lower().split())
                palavras_resposta = set(resposta.lower().split())
                palavras_corretas = palavras_esperadas & palavras_resposta
                
                if len(palavras_corretas) / len(palavras_esperadas) >= 0.5:  #Treshhold de 50%
                    respostas_corretas += 1
                total_perguntas += 1
    
    acuracia = (respostas_corretas / total_perguntas) * 100
    return acuracia

if __name__ == "__main__":
    file_path = "testes.txt"
    acuracia = calcular_acuracia(file_path)
    print(f"A acurácia do chatbot é: {acuracia:.2f}%")