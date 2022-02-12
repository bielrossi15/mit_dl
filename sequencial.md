# Deep Sequencial Modeling

## Dados sequenciais
- Audio pode ser separado em uma sequencia de ondas mecanicas;
- Textos podem ser separados em sequencias de caracteres ou palavras, podendo ser tratados como passos no tempo da nossa sequencia;
- Predicao de precos da bolsa;
- Genoma;
- etc;

## Aplicacoes de modelos sequenciais
- Processamento de linguagem natural, tratando de palavras como etapas no tempo;
    - EX: Classificacao de sentimentos, **MANY TO ONE**;
    - Legenda de imagem, **ONE TO MANY**;
    - Traducao de linguagem, **MANY TO MANY**;
        - Many to One: sequencia de entradas e uma saida;
        - One to Many: uma entrada para uma sequencia de saidas;
        - Many to Many: sequencia de entradas e sequencia de saidas;

## Neuronios com recorrencia
- xt -> |_| -> y_t;
- xt eh o x0, x1, x2, etc, que eh a entrada de um ponto especifico no tempo, sendo y_hat t sua predicao;
- |_| eh uma camada da rede neural;
- y_t = f(xt), entao isso quer dizer que a saida depende apenas do xt se tratarmos como dados isolados;
- deve haver uma memoria para conectar as diferentes entradas de tempo, poder passar a informacao para os dados futuros da sequencia;

   y_0        y_1             y_n
    |    h0    |         hn    |
   |_|  --->  |_|   ... --->  |_|
    |          |               |
    x0        x_1              xn
- Criando essa relacao de recorrencia, cria-se uma memoria;
- Entao y_t = f(xt, ht-1), a funcao depende da sua entrada temporal e tambem da sua memoria passada;
- Pode ser visto tambem como um ciclo;
- Sao chamadas de RNN (RECURSIVE NEURAL NETWORKS);

## RNN'S
- Mantem um estado interno, ht, que eh atualizado a cada passo temporal, enquanto a sequencia eh processada;
- Aplica-se uma relacao de recorrencia a cada passo temporal para processar a sequencia, sendo ele:
    - ht = fw(xt, ht-1);
        - ht: estado da celula;
        - fw: funcao usando os pesos W;
        - xt: input;
        - ht-1: estado antigo;
    - Parametrizada por um set de parametros (pesos);
- A mesma funcao e o mesmo set de parametros (pesos) sao usados a cada passo temporal;
- Intuicao:
    - Inicializar a rnn, o ht (hidden state) e a frase;
    - Fazer o loop entre as palavras na rnn e enviar a palavra e o ht anterior na rnn;
    - Isso gera uma predicao para a proxima palavra e uma atualizacao para o ht;
    - No fim, a rnn gera um input final;
    ```
    rnn = RNN()
    hidden_state = [0, 0, 0, 0]

    sentence = ["I", "love", "recurrent", "neural"]

    for word in sentence:
        prediction, hidden_state = rnn(word, hidden_state)

    nex_word_prediction = prediction
    ```

## Matematica da RNN
- vetor de entrada xt;

xt --> ht<>[]<>ht --> y_t; (ht eh usado dentro da rnn)

- atualizar o ht:
    - ht = tanh(W1 * ht-1 + W2 * xt);
    - tanh pode ser qualquer funcao de ativacao;
    - W1 = Wt(hh), matriz de peso do ht;
    - W2 = Wt(xh), matriz de peso da entrada;
- vetor de saida no tempo t:
    - y_t = W3 * ht (normalmente y_t = ht);
    - W3 = Wt(hy), matriz de peso diferente das anteriores;
- Representacao:
  
                **L**
    /             |                 \
    L0            L1                Lt
    ^             ^                  ^
    |             |                  |
   y_0           y_1                y_n
    ^             ^                  ^
    |             |                  | 
    W(hy)         W(hy)              W(hy)
    |             |                  | 
    _    W(hy)    _        W(hy)     _ 
   |_|  ------>  |_|   ... ------>  |_|

    |             |                  |
    W(xh)         W(xh)              W(xh)  
    |             |                  | 
    ^             ^                  ^
    x0           x_1                 xn

- **IMPORTANTE**: **UTILIZAR AS MESMAS MATRIZES DE PESO A CADA PASSO TEMPORAL**;

```
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self.rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init()__()

        # initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dims])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_yh = self.add_weight([output_dim, rnn_units])

        # initialize ht to zero
        self.h = tf.zeros([rnn_units, 1])
    
    def call(self, x):
        # forward pass

        # update ht
        self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)

        # compute output
        output = self.W_hy * self.h

        # Return current hidden state
        return output, self.h
```
OU
```
tf.keras.layers.SimpleRNN(rnn_units)
```
## Criterios de Design
- Handle variable-lenght sequences;
- Track long-term dependencies;
- Maintain information about order;
- Share parameter across the sequence (weight sharing);
- RNN's meet this model design criteria;

## EX: Predicao de palavras
- Transformar palavras em vetores/arrays, embedding:
    - Transformar indicies em vetores de tamanho fixo;
    - Mapear toda palavra que esta/pode estar no vocabulario a ser usado/predito;
    - Transformar as palavras em indicies;
    - One-hot embedding/Learned embedding;

## Backpropagation RNN (BPTT - Backpropagation Throug Time)
- Os erros sao propagados da funcao de Custo ate os passos no tempo e depois do ultimo ate o primeiro passo temporal;

## Problemas de gradiente
- Computar o gradiente baseado no estado inicial h0 envolve muitos fatores de W(hh) e muita computacao do gradiente, levando aos seguintes problemas:
    - Muitos valores > 1: Gradientes explodindo, impossivel conseguir otimizar pois os gradientes ficam muito grandes, tendo como solucao **Gradient clipping**, escalar os valores dos gradientes;
    - Muitos valores < 1: Gradientes sumindo, gradientes ficam pequenos demais, se tornando impossivel treinar a rede neural, pode-se usar:
        - Escolha de funcao de ativacao correta;
        - Inicializacao de pesos;
        - Arquitetura;

## Vanishing Gradients
- Multiplicar numeros pequenos por outros numeros pequenos torna cada vez mais dificil propagar erros;
- Rede fica enviezada a capturar dependencias de curto prazo e nao de longo prazo;
- Normalmente, RNN's ficam cada vez menos sensiveis a memoria de longo prazo por conta desse problema em especifico;
- Nao podem aprender padroes muito antigos;
- Podemos usar:
    - Escolha de funcao de ativacao: 
        - ReLU: impede que f' diminua o gradiente quando x > 0, pois f' com x>0 = 1;
    - Inicializacao de parametros:
        - Inicializar pesos com a matriz identidade;
    - Usar unidades recorrentes mais complexas, com "portoes" para controlar a informacao que eh passada;

## Celulas com Portoes (LSTM, GRU, etc)
- LSTM (long short term memory);
- Melhores a guardar informacoes antigas, passadas por celulas anteriormente e acompanhar informacao entre varios passos temporais;

## LSTM
- RNN normal:
                     ht
                      ^
ht-1 --->             |
          [TANH] ---------> ht
xt   --->

- Linhas representam as multiplicacoes das matrizes de peso e [TANH] eh a funcao de ativacao;
- Modulo repetivel, um no computavel da RNN;

- LSTM:
    - ct = memoria longo prazo;
    - ht = memoria de curto prazo;
- Celula LSTM:

                                                            ht
           ft                        ct                     ^
ct-1 --->  (x)------cft------(+)-----------------|----------|---> ct
            |             it  |              ot[tanh]       |
          [sig]        ------(x)            ----(x)         |
ht-1 --->   |         |       |            |     |          |
             -------[sig]---[tanh]-------[sig]    --------------> ht
xt   --->   |

- xt: input;
- [sig]/[tanh]: dense layers que representam funcoes de ativacao;
- conexao ht-1->ht: memoria de curto prazo, pois pega a saida do ultimo noh;
- ct (cell state): memoria de longo prazo, apenas atualizada duas vezes (x)(+), menos computacoes -> gradiente mais estavel;
    - (x): elementwise multiplication;
    - (+): elementwise summation;

- **ft (FORGET) (x): decide o que esquecer na memoria de longo prazo**;
    - *matriz ft = sigmoid(Wf.[ht-1,xt] + bf) ou sigmoid(xt * Uf + ht-1 * Wf)*;
        - Contatenamos ht-1 e xt, multiplicamos por uma matriz de peso e adicionamos um bias, isso resulta numa matriz de filtro;
        - Wf e Uf: matriz(es) forget;
        - Sigmoid possui o intervalo (0, 1), entao quanto mais proximo de zero, esquecemos mais;
    - *C(f,t) eh o resultado da multiplicacao (x), C(f,t) = Ct-1 (x) ft, que eh a aplicacao do filtro*;
        - C(f,t) eh a celula do passo temporal passado e decidimos nesse passo atual o que esquecer;
        - EX:
            ``` 
            Ct-1 = [1,2,4]
            ft = [1,0,1]
            C(f,t) = [1,2,4] (x) [1,0,1] = [1,0,4]
            ```
        - Aqui escolhemos o que lembrar e o que esquecer;

- **it (INPUT) (x)(+): decide o que eh importante do input**;
    - *it = sigmoid(Wi.[ht-1,xt] + bi) ou sigmoid(xt * Ui + ht-1 * Wi)*;
    - *Ct', nova ct (cell state), tanh(Wc.[ht-1,xt] + bC) ou tanh(xt * Uc + ht-1 * Wc)*;
    - *C(i,t) eh o resultado da multiplicacao C(i,t) = Ct' (x) it, que retorna a matriz com o que eh relevante do input*;

- **ct (Cell state) (+): novo cell state que vai ser passado pra frente, soma o que deve ser lembrado C(f,t) com o que deve ser adicionado C(i,t)**:
    - *ct = C(f,t) (+) C(i,t), basicamente eh o novo ct (cell state) que diz que informacao devemos lembrar C(f,t) e o que devemos adicionar como informacao nova C(i,t)*;

- **ot (OUTPUT) (+): saida da matriz**;
    - *ot = sigmoid(Wo.[ht-1,xt] + bo) ou sigmoid(xt * Uo + ht-1 * Wo)*;
    - *ht = ot (x) tanh(Ct)*:
        - Utilizamos tanh(Ct) para adicionar nao-linearidade na rede neural e normaliza os valores;

## Dimensões
- Exemplo: imagens de 30x15, quais as dimensoes?
    - Number of sequences: Quantidade de sequencias, por exemplo, quantidade de fotos, quantidade de frases, etc;
    - Time steps: 30, pode ser visto como a quantidade de vezes que a célula vai processar diferentes pixels ou qtd de palavras em uma frase;
    - Input sequence lenght: 15 (qtd de pixels por linha) ou em texto, tamanho de cada palavra após processamento;
    - Output dim: Qtd de células, pois a camada de RNN possui várias células e oq vemos como várias células são apenas passos temporais da mesma célula;
     ``` 
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(time_steps (30), input_sequence_length (15) )))
            model.add(tf.keras.LSTM(number_of_units))
     ```
     - Não é necessário explicitar o número de sequencias pois isso vem diretamente do dataset;
    

