# Uso de Inteligência Artificial Generativa

Ferramentas Utilizadas: OpenAI ChatGPT (GPT-4o), Anthropic Claude 3.5 Sonnet, Google Gemini 1.5 Pro  

---
O uso de ferramentas de IA Generativa neste projeto limitou-se ao suporte de programação (debugging, otimização de funções vetorizadas), estruturação de relatórios e documentação, e brainstorming estatístico para diagnóstico de resíduos. Nenhuma decisão analítica, interpretação de parâmetros ou aprovação final de código foi realizada de forma autônoma pelas ferramentas de IA sem a devida validação técnica e execução local pelos integrantes do grupo.

---

## Registro de Interações

### Interpretação de ACF e PACF

Usamos a ferramenta para revisar como padrões observados nesses gráficos poderiam orientar a escolha inicial das ordens de um modelo ARIMA/SARIMA, especialmente considerando a sazonalidade semanal de período 7.

### Revisão da implementação do MASE

Pedimos ajuda para conferir se a escala usada no denominador correspondia ao erro in-sample do naive sazonal semanal calculado apenas sobre o conjunto de treino.

### Debugging pontual

Em alguns momentos, enviamos pequenos trechos de código e mensagens de erro para identificar problemas de índices, dimensões de arrays/dataframes e alinhamento entre datas observadas e previsões.

### Revisão do split temporal

Usamos a IA para conferir a lógica de separação entre treino e validação e verificar se algum trecho poderia introduzir vazamento de informação.

### Comentários e documentação

As ferramentas foram usadas para sugerir formas mais claras de comentar algumas funções e explicar etapas do pipeline no notebook/README.

## Prompts representativos

#### "estou analisando uma série temporal diária de vendas com sazonalidade semanal. na ACF aparecem picos fortes nos lags 7, 14 e 21. como isso pode orientar a escolha inicial dos termos sazonais de um SARIMA? não escreva o código completo; explique apenas como interpretar os gráficos."

#### "para calcular MASE com sazonalidade 7, estou usando no denominador o MAE entre y[t] e y[t-7] somente no conjunto de treino. depois divido o MAE da previsão na validação por esse valor. essa implementação corresponde à definição de MASE sazonal? há algum risco de leakage nessa lógica?"

#### "esse trecho tá retornando previsões com 29 linhas enquanto minha validação tem 28 dias. pode me ajudar a identificar o possível erro de índice ou intervalo? não reescreva o script inteiro; indique onde está o problema e como corrigir."

## Erros da IA que o grupo corrigiu

### Cálculo incorreto da escala do MASE

Em uma das respostas, a IA sugeriu calcular o denominador do MASE usando diferenças consecutivas (y[t] - y[t-1]), que seria adequado para uma escala baseada em naive não sazonal. Como as séries desta tarefa têm sazonalidade semanal e o enunciado determina período 7, corrigimos o cálculo para usar os erros in-sample do naive sazonal:
|y[t] - y[t-7]|
calculados exclusivamente no conjunto de treino.

### Sugestão de refazer o modelo após observar a validação

Ao discutir um resultado ruim do SARIMA, a IA sugeriu testar novas especificações usando diretamente o desempenho nos 28 dias de validação como critério repetido de escolha. Consideramos que isso poderia levar a um ajuste excessivo à validação. Em vez disso, mantivemos a identificação baseada no treino, nos diagnósticos da série e em critérios de ajuste apropriados, usando a validação principalmente para a comparação final entre os modelos.

### Desalinhamento de datas no debugging

Em uma sugestão de correção de código, a IA propôs construir o intervalo de previsão incluindo também o último dia do treino, o que gerava uma previsão a mais e deslocava o alinhamento com a validação. O grupo identificou o problema ao conferir o número de linhas e as datas esperadas e corrigiu o intervalo para conter exatamente os 28 dias de 2016-03-28 a 2016-04-24.


## Responsabilidade

A responsabilidade pelas decisões metodológicas, pela validação dos resultados
e pela versão final do código permaneceu integralmente com os integrantes do
grupo. As respostas geradas pelas ferramentas de IA foram revisadas e testadas
antes de qualquer incorporação ao projeto.
