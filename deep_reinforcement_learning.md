# Deep Reinforcement Learning

## Reinforcement Learning
- Robotica;
- Jogos;

## Classes de aprendizado
- Supervised:
    - Data (x,y): x dados, y label;
    - Objetivo: aprender uma funcao para mapear x -> y;
    - EX da laranja:
        "Isso eh uma laranja";
- Unsupervised:
    - Data (x): x dados, sem label;
    - Objetivo: aprender uma estrutura interna/oculta;
    - EX da laranja:
        "Isso (laranja) eh parecido com essa outra coisa (laranja)";
- Reinforcement:
    - Data: pares de estado(observacoes do sistema)-acao(o que o agente faz quando observa esse estado);
    - Objetivo: maximizar recompensas no futuro apos passos temporais;
    - EX da laranja:
        "Coma isso pois isso te mantera vivo";

## Conceitos-chave
- Agente: quem toma as acoes;
- Ambiente: aonde o agente se encontra;
- Acao: um movimento que o agente pode fazer no ambiente, sendo:
    - Espaco de acao A: set de acoes que um agente pode fazer no ambiente;
    - at: acao do agente no tempo t;
- Observacoes: como o ambiente interage de volta com o agente, feedback, e como suas acoes afetaram seu estado dentro do ambiente sendo:
    - estado: uma situacao concreta e imediata em que o agente se encontra;
    - recompensa: feedback de como o ambiente mede o sucesso ou falha das acoes do agente, podem ser imediatas ou com um delay, avaliacao das acoes do agente;
    - recompensa total: Rt = SUM(ri) = rt + rt+1 + rt+2 + ... + rt+n + ...;
    - recompensa total com desconto: Rt = SUM(gamai*ri), criado para afetar as escolhas do agente, desenhada para fazer recompensas futuras serem menos importantes do que as imediatas;
        - gama: fator de desconto, 0 < gama < 1;
    
        recompensa rt
     mudanca de estado st+1
    <-----------------------
AGENTE                    AMBIENTE
    |                       |
    ----------------------->
            acoes
            acao at

## Q-function
- **Rt = rt + gama * rt+1 + gama^2 * rt+2 + ...**;
- Recompensa total, Rt eh a soma descontada de todas as recompensas obtidas a partir do tempo t;
- **Q(st,at) = E[Rt|st,at]**;
    - Entrada: estado que o agente se encontra e a acao que o agente toma nesse estado;
    - Retorna a recompensa esperada total futura;
    - E[X|Y]: valor esperado da variavel aleatoria X dado Y;
    - O agente precisa escolher a melhor acao a ser tomada, a que retorna maior recompensa;
- Estrategia: o agente precisa de uma politica pi(s) que escolhe uma acao para maximizar a recompensa no estado atual;
    - **pi(s) = argmax Q(s,a)**; 
                 **a**

## Algoritmos de Deep Reinforcement Learning 
- Value learning: 
    - Ache Q(s,a);
    - a = argmax a Q(s,a)
- Policy learning, mais direto, sem aprender Q:
    - Find pi(s);
    - a ~ pi(s);

# Deep Q Networks (Value learning)
- Como usar redes neurais para modelar q-functions?
- acao + estado -> retorno esperado;

estado,s ---->  
                |             |
                | REDE NEURAL | -> Q(s,a)
                |             |
acao, a ----->  

- Modelo ineficiente, pois com esse modelo devemos computar a rede para cada passo temporal;
- Modelo mais conveniente:
- estado -> retorno esperado para cada acao;

                |             |     Q(s,a1)
estado,s ---->  | REDE NEURAL | ->  Q(s,a2)
                |             |     ...
                                    Q(s,an)
- Como treinar?
    - Qual eh o melhor caso? Como o agente atuaria de uma maneira ideal? Tomar as melhores acoes?;
    - Se isso acontecesse, o target return seria maximizado, entao isso eh treinar o agente;
    - **Target = r + gama * argmax_a'(Q(s',a'))**:
        - r: recompensa inicial;
        - gama: fator de desconto;
        - argmax_a'(Q(s',a')): acao selecionada que maximiza o retorno esperado do estado futuro;
    - Predito pela rede: Q(s,a);
    - **Custo = E[||(r + gama * argmax_a'(Q(s',a'))) - Q(s,a) ||^2]**:
        - MSE entre o target e o que foi predito pela rede neural;

## Resumo
- Usar rede neural para aprender Q-function e usar para inferir a funcao pi(s) otimizada;
- Escolhemos a acao que resulta no maior valor para a Q-function e retornamos ele para o estado;
- Recebemos o novo estado;

## Desvantagens
- Complexidade:
    - Pode modelar cenarios aonde o espaco de acoes discretos e pequenos;
    - Nao consegue lidar com espacos continuos;
- Flexibilidade:
    - pi(s) (politica) eh deterministicamente computada pela Q-function por maximizacao da recompensa, entao nao pode aprender politicas estocasticas;

# Policy lerning

## Diferenca de DQNs
- Ao inves de aprender a funcao Q para inferir a politica otima, aprendemos diretamente a politica pi(s);
- A saida eh muito mais direta e vai dar a maior recompensa;
- Prediz uma distribuicao probabilistica;
- Para predizer a acao, apenas retiramos uma amostra dela;
- Por conta disso, podemos pegar amostras diferentes por nao terem probabilidade zero;

                |             |     P(a1|s)
estado,s ---->  | REDE NEURAL | ->  P(a2|s)
                |             |     ...
                                    P(an|s)

## Vantagens
- Muito mais direta, otimizando diretamente a politica;
- Pode ser usada em espacos de acoes continuos, pois retorna uma distribuicao probabilista que tem valores definidos;
- Podemos visualizar isso como um espaco de acao continuo, usando qualquer distribuicao probabilistica, normalmente a gaussiana;
- O ponto aonde ela eh maior, diz quanto ela tem que andar, ao inves de so para onde deve andar;

## PG (policy gradient):
- Permite modelagens de espacos de acao continuos;

                |             |     Media, mu
estado,s ---->  | REDE NEURAL | ->  
                |             |     Variancia, sigma^2

- Predicao agora eh uma distribuicao probabilistica normal gaussiana, pois temos um numero infinito de acoes, entao predizer a probabilidade de cada acao seria impossivel;
- P(a|s) = Normal(mu, sigma^2);
- Se predizemos uma media -1 e uma variancia 0.5, retiramos a amostra naquela posicao e temos o valor continuo;
- Nao estamos presos ao valor maior, pois eh uma distribuicao probabilistica continua;
- Integral de P(a|s) = 1;

## Treinar o algoritmo
- 1: Inicializar o agente;
- 2: Rodar a policy ate o termino;
- 3: Gravar todos os estados, acoes e recompensas;
- 4: Diminuir probabilidades de acoes que resultaram em baixa recompensa;
- 5: Aumentar a probabilidade de acoes que resultam em alta recompensa;

## Como diminuir/aumentar probabilidades
- **loss = -log P(at|st) * Rt**:
    - -log P(at|st): log likelihood de uma acao;
    - Se uma acao tem uma recompensa alta e uma log likelihood alta, entao essa acao seria reforcada;
    - Se uma recompensa eh baixa e uma log likelihood alta, entao ajustaria essa probabilidade pra essa acao nao ser usada no futuro;

- Gradient descent:
    - **w' = w - gradient(loss)** que eh igual a:
        - **w' = w + gradient(log P(at|st) * Rt)**;
