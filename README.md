# ChatBot vestibular Unicamp

ChatBot criado para o processo seletivo do estágio na NeuralMind Julho 2024

---
# Instalações necessárias 

Para instalar o chatbot comece clonando o repositório  
Adicione a Chave da API da OpenAI e do LlamaCloud ao arquivo .env  
Em seguida instale as dependencias necessárias com:  
```bash
pip install -r requirements.txt  
```

---
# Execução

Adicione suas chaves OpenAi e LlamaCloud no arquivo .env  
Para rodar o chatbot execute:  
```bash
streamlit run main.py  
```
Para rodar o teste automatizado:  
```bash
python3 teste_acuracia.py  
```

---

# Considerações iniciais e entrada de dados

Em primeiro momento busquei entender o funcionamento geral de um llm,e ao pesquisar notei que o desenvolvimento se basearia em 3 passos:Extração dos dados da página com as informações sobre o vestibular,embbedding dos dados(transformação em representação vetorial),e alimentação de um modelo de chat com os dados após realizar o embbediing.

---

# Desenvolvimento

Para o desenvolvimento utilizei a biblioteca llama_index e a API da OpenAI integrada ao llama_index.Em primeiro momento foi feita a leitura dos dados txt sobre o vestibular com o recém lançado LlamaParse que permite a leitura de arquivos de variados formados,testei também com as informações no formato pdf e os resultados foram similares.Em seguida os dados foram transformei em um indice vetorial com auxilo do OpenAiEmbbedings e esse indice vetorial transformado em um motor de consulta passando um chat OpenAi 3-5 turbo como parametro.Por fim,com auxilio do streamlit criei uma interface para receber a dúvida do usuario,passar ao modelo no formato de uma query e exibir a resposta ao usuário.

---

# Observações e possíveis melhorias

Observei que o chat possui pouco poder de inferência da pergunta do usuario,ou seja perguntas mal formuladas resultam em respostas ruins.Acredito que um modelo que fosse armazenando as perguntas e as respostas feitas pelo usuario iria ter melhor poder de inferência sobre as perguntas e realizar pesquisas melhores e por fim formular respostas melhores.Como esse é um modelo que usa uma base de dados fixa provinda do arquivo de texto com as informações sobre o vestibular,acredito que seja esse o motivo da limitação.Sendo a implementação de uma base de dados que pudesse armazenar essas perguntas uma possível melhoria para uma futura versão.

---

# Testagem e resultados obtidos

O desafio seguinte foi achar uma maneira de testar o modelo,uma vez que não existe resposta fechada para a maioria das perguntas e ate para as quais a resposta é fechada,essa resposta pode ser apresentada de varias maneiras.Para resolver esse problema criei um arquivo de teste que itera sobre um arquivo txt com perguntas e respostas baseadas nas informações sobre o vestibular,comparando o conjunto de palavras da resposta obtida pelo modelo com a resposta contida no arquivo de informações.Se a o tamanho do conjunto obtido pela interseção da resposta obtida com a resposta real,divido pelo tamanho da resposta real for maior do que um treshhold(nesse caso definido como 0,5) a resposta é considerada satisfatória.E por fim calcula a porcentagem de respostas consideradas satisfatórias.

Vale ressaltar que essa maneira de analisar a acertividade do modelo fica restrita as perguntas que eu fiz,se as perguntas fossem outras o resultado seria outro.Mas como a ideia principal era ter uma noção de como o bot estava funcionando,considerei válida essa maneira de testar.

No final dos testes a acurácia do bot foi 0.7111 ou 71.11%,o que considerei um bom resultado.

---

