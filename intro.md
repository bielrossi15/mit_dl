# Intro to Deep Learning

## O que eh dl
- IA: qualquer tecnica que permite computadores imitar comportamento humano;
- Machine Learning: Habilidade de aprender sem ser programado explicitamente;
- Deep Learning: Extrair padroes de dados usando redes neurais;

## Perceptron
- Forward prop: 
    - y_hat = g(w0 + SUM(xi * wi)):
    - y_hat = saida;
    - g = funcao nao linear;
    - w0 = bias (para fazer um shift nas saidas;
    - SUM = soma de i ate M;
    - xi = entradas;
    - wi = pesos;
    - xi * wi = combinacao linear das entradas;

    **PODE SER REPRESENTADA TAMBEM POR**
    - y_hat = g(w0 + XtW);
    - X = [x1...xm];
    - W = [w1..wm];
    - produto escalar dos dois vetores + bias dentro de uma funcao nao linear g;

- Funcoes nao lineares:
    - Essas funcoes sao necessarias pra introduzir nao-linearidade a rede neural, pois os dados normalmente sao nao lineares;
    - sigmoid = 1/(1 + e**(-z)), [0,1]; 
    `tf.math.sigmoid(z)`
    - ReLU = g(z) = max(0,z), [0, 1];
    `tf.nn.relu(z)`

## Criando redes neurais com perceptrons
- z = w0 + XtW ou w0 + SUM(xjwj) perceptron simples;
- zi = w0,i + SUM(xjwj,i) perceptron com varias saidas, pode ser chamado de **dense layers**;
- Dense layers: Camadas altamente conectadas, pois todas as entradas estao conectadas em todas as saidas;
- Implementacao:

```
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        z = tf.matmul(inputs, self.W) + self.b
        output = tf.math.sigmoid(z)
        
        return output    
```
ou
```
import tensorflow as tf

layer = tf.keras.layers.Dense(units=2)
```

## Forward prop NN
- Possui uma camada chamada Hidden Layer;
- Cada layer vai ter uma propria matriz de pesos W;
```
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])
```
- Para criar uma DNN, precisamos apenas colocar mais hidden layers e conecta-las;
- zk,i = w(k)0,i + SUM j to nk-1 (g(zk-1,j) w(k)j,i), sendo:
    - z: valores dos "neuronios";
    - k: numero da camada em que se encontra;
    - i: posicao do neuronio dentro de sua camada;
    - j: posicao do neuronio da camada anterior;
    - g: funcao de ativacao; 
    - w(k): matriz de dimensao n[l] x n[l-1] que contem os pesos;
    - n[l]: quantidade de neuronios na camada atual;
    - n[l-1]: quantidade de neuronios da camada anterior;
    
    ou
- z(1) = X*W(1);
- a(k) = g(z(k));
- z(k+1) = a(k)W(k);
- a(k+1) = g(z(k+1)):
    - X: entradas;
    - z: vetor com os valores da camada anterior;
    - a: resultado da funcao de ativacao aplicada nos valores anteriores;
    - z(k+1): valores dos neuronios anteriores com o produto escalar feito com a matriz dos pesos que ligam os neuronios anteriores a ele;
```
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n1)
    .
    .
    .
    tf.keras.layers.Dense(nL)
])
```

## Loss e Empirical loss (Cost)
- Loss: Funcao que calcula a incorretude das predicoes em cada valor;
- Empirical loss (Cost): a media de todas as funcoes de "loss" que tivemos;
- Existem tipos diferentes de funcoes de Cost/Empirical loss:
    - softmax cross entropy loss 
    ` loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,predicted)) `
    - MSE
    ```
    loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted)))
    loss = tf.keras.losses.MSE(y, predicted)
    ```

## Treinando ANN's/Gradient descent
- Queremos achar os pesos que tenham o menor custo possivel:
    W* = argmin J(W)
    W = {W(0), W(1)...}
    J = Cost
- Gradient descent:
    - Inicializar os pesos randomicamente;
    - Loop ate convergencia:
        - Computar gradiente, d_parcialJ(W)/d_parcialW;
        - Atualizar pesos, W <- W - lr d_parcialJ(W)/d_parcialW;
    - Retornar pesos;
    - lr = learning rate;

```
import tensorflow as tf

weights = tf.Variable([tf.random.normal()])

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)
    
        weights -= lr * gradient
```

## Backpropagation
- Como computar o gradiente;
- Gradiente diz a direcao que devemos ir para chegar ao minimo "global";
- A cada gradiente calculado, ele vai diminuir para nao dar um passo maior do que deve e passar do minimo;
- Gradiente = dp J(W)/dp W;

     w1             w2 
x ---------> z1 -----------> y_h -> j(W)

dp J(W)/dp w2 = (dp J(W)/dp y_h) * (dp y_h/dp w2) 
dp J(W)/dp w1 = (dp J(W)/dp y_h) * (dp y_h/ dp z1) * (dp z1/dp w1)
- Funciona aplicando recursivamente a regra da cadeia para chegar de novo no comeco e treinar de novo;
- A aplicacao do gradiente diz o quanto a funcao de custo eh sensivel pra cada peso e cada bias, qual a importancia dele pra a nn;
- A regra da cadeia eh aplicada pois eh necessario matematicamente, mas se for para entender logicamente, acontece pois a mudanca em w(2) afeta tambem o neuronio, causando uma pequena mudanca (dp y_h/dp w2), e isso tras a necessidade de computar tudo para conseguir ter a derivada dp J(W)/dp w2, ou quanto a mudanca de w2 vai afetar J(W);
- dp J(W)/dp w2 = (1/n) SUM(dp J(W)/dp w2), isso significa que a derivada final eh a media de todas as derivadas de todos os exemplos de treino nesse neuronio;
- Ela eh usada recursivamente pois depois queremos saber o quanto cada camada afetara cada camada recursivamente;

## Learning rate
- Learning rate muito pequena muito devagar e tambem pode ficar preso em minimos locais;
- Learning rate muito grande pode divergir;
- Algoritmos de otimizacao ajudam:
    - SGD;
    - ADAM
    - ADADELTA;
    - ADAGRAD;
    - RMSPROP;

```
import tensorflow as tf

model = tf.keras.Sequential([...])

optimizer = tf.keras.optimizer.SGD()

while True:
    prediction = model(x)

    with tf.GradientTape() as tape:
        loss = compute_loss(y, prediction)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

```

## Mini Batches
- Ao inves de usar o dataset inteiro, escolhemos uma parte apenas e fazemos todo o gradient descent;

## Overfitting
- Dropout: randomicamente setar algumas funcoes de ativacao para zero;
- Early stopping