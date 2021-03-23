# Deep Computer Vision

## Visao Computacional
- Detectacao/Reconhecimento facial;
- Imagens de exames: identificar doencas;
- Carros autonomos;
- Acessibilidade;

## O que computadores veem?
- Imagens sao numeros (listas 2D de numeros);

## Tipos de algoritmos
- Regressao: continuo;
- Classificacao: variavel de saida prediz uma label de alguma classe;
- Como?
    - Classificacao: Identificando features chave em cada categora, ex:
        - Ser humano: nariz, olhos, boca, etc;
        - Carros: placa, lanterna, rodas, etc;
        - Casas: portas, janelas, etc;

## Extracao de features manualmente
- Problemas de cada passo:
    - Domain knowlege: viewpoint variation, illumination conditions, etc;
    - Define features: scale variation, deformation, background clutter, etc;
    - Detect features to classify: occlusion, intra-class variation, etc;

## Aprender representacoes de features
- Eh possivel aprender a hierarquia de features diretamente dos dados ao inves de manualmente?
    - Features de baixo nivel: margens, linhas, pontos escuros, etc;
    - Features de nivel medio: olhos, ouvidos, narizes, etc;
    - Features de alto nivell: estrutura facial;

## Limites Rede Neural Padrao
- Entrada: matriz 2D transformada num vetor de uma dimensao;
- Por conta disso, toda a informacao espacial eh perdida (pixels que sao vizinhos, etc);
- Muitos parametros;

## Usando Estrutura Espacial
- Separar as matrizes em varias matrizes de menor area e passar elas para a rede;
- Isso diminui a quantidade de pesos;

## Componentes CNN
- Convolution;
- Pooling;

# Convolution
- Filtro (filter): grid de pesos;
- "Aplicar" o Filtro na imagem;
- O resultado eh um feature map, que eh um vetor 2D de mesmo tamanho da entrada e baseado no filtro, que vai, basicamente conter as features, para qual o filtro foi treinado, extraidas;
    - EX: Filtro detecta linhas verticais, feature map do filtro sera um feature map dizendo em que locais foi detectado linhas verticais;
    - Como ocorre a convolucao:

        imagem            Filtro 
    | 1 2 3 4 5 6 |     | 1 2 3 |
    | 1 2 3 4 5 6 |     | 1 2 3 |
    | 1 2 3 4 5 6 |     | 1 2 3 | 3x3
    | 1 2 3 4 5 6 |
    | 1 2 3 4 5 6 |
    | 1 2 3 4 5 6 | 6x6

          saida           
    | . . . . . . |    
    | . ? ? ? ? . |    
    | . ? ? ? ? . |     
    | . ? ? ? ? . |
    | . ? ? ? ? . |
    | . . . . . . | 6x6

    - Coloca o centro do Filtro em um quadrado do mesmo tamanho dele na imagem (por exemplo, o quadrado superior 3x3);
    - Calcula o produto escalar SUM(imagem(i) . Filtro(i)) e substitui o valor q foi centrado na imagem pelo resultado do produto escalar;
    - Se existir posicoes em que nao eh possivel alinhar o centro sem ficar partes do Filtro de fora, por exemplo, nas extremidades dessa imagem (valores substituidos por ponto), normalmente adicionamos uma camada de zeros por fora de tudo, ex:

    | 0 0 0 0 0 0 0 0 |    
    | 0 1 2 3 4 5 6 0 |    
    | 0 1 2 3 4 5 6 0 |     
    | 0 1 2 3 4 5 6 0 |
    | 0 1 2 3 4 5 6 0 |
    | 0 0 0 0 0 0 0 0 | 8x8

    - Assim, a imagem original (aonde nao possui o zero) vai poder, em todos os pixels, aplicar o Filtro;

## Filtro
- Mas o que eh o Filtro (filter)?
- Detector de features;
- Filtros sao aprendidos pela rede neural;
- Ex:
    - Filtro identificador de linhas verticais:

        | 0 1 0 |
        | 0 1 0 |
        | 0 1 0 | 3x3

    - Filtro identificador de linhas obliquas:
        
        | 1 0 0 |
        | 0 1 0 |
        | 0 0 1 | 3x3

## Decisoes Arquiteturais para Convolucao
- Tamanho do grid;
- Stride;
- Profundidade;
- Numero de Filtros;


## Tamanho do Grid
- Numero de pixels altura/largura;
- Normalmente numeros impares, pois tem valores centrais;

## Stride
- Tamanho do passo usado para fazer o slide do Filtro na imagem;
- Indicado por pixels;

## Profundidade
- Imagens em grayscale, igual a 1;
- Em RGB, igual a 3;

## Numero de Filtros
- Uma camada de convolution tem multiplos Filtros;
- Cada Filtro tem uma saida de um array 2D;
- Saidas de cada camada tem o mesmo tanto de arrays 2D e numero de Filtros;

# Pooling
- Diminuir o tamanho de uma imagem;
- Parecido com a convolution, passando um grid por cima de uma imagem;
- Max/Average pooling, normalmente utilizado o Max;
- Nao tem parametros, entao nao eh aprendido nada com ele;

       imagem         saida              saida
    | 1 2 3 4 |      | ? ? |            | 2 4 |
    | 1 2 3 4 |      | ? ? | 2x2   ->   | 2 4 | 2x2
    | 1 2 3 4 |      
    | 1 2 3 4 | 4x4

- Separa uma matriz no canto superior esquerdo da imagem do tamanho da saida e acha o valor maximo (ou media, depende do tipo de pooling), coloca esse valor na saida e vai o tanto de stride para o lado;

## Decisoes Arquiteturais;
- Tamanho do grid;
- Stride;
- Tipo;

# CNN
- Entrada -> Convolucao -> Nao-lineariedade -> pooling;
- Convolution vai, basicamente, aplicar varios filtros que serao aprendidos pela rede neural, e ter como saida uma quantidade de feature maps, que sao do tamanho da imagem de entrada, igual a quatidade de filtros
- Convolution:
    - Code: `tf.keras.layers.Conv2D(filters=f, kernel_size=(h,w), strides=s)`;
    - Settings: quantidade de filtros, tamanho do kernel e strides;
    - Saida de cada camada de convolucao vai ser: h x w x f;
        - h: Altura da imagem de entrada;
        - w: Largura da imagem;
        - f: Quantidade de filtros;
- Nao-linearidade:
    - Code: `tf.keras.layers.ReLU`;
    - Normalmente usando ReLU, se x < 0, x = 0;
- Pooling:
    - Code: `tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)` ou average pooling;
    - Settings: tamanho do pool, strides;
    - Reduzir a dimensao das imagens;
- Apos isso, o resultado do pooling vai para a proxima camada e assim segue;
- Features: low-level sendo compostas em mid-level sendo compostas em high-level;
- Entao uma rede neural completa para classificacao de imagens fica assim:
    - INPUT -> CONV+RELU -> POOLING -> CONV+RELU -> POOLING -> ... -> FLATTEN -> FULLY-CONNECTED -> SOFTMAX;
    - Aprender as features nas imagens utilizando **CONVOLUTION**;
    - Introduzir **nao-lineariedade** com funcoes de ativacao, pois dados reais normalmente sao nao-lineares;
    - Reduzir a dimensao e preservar a invariacao espacial com **POOLING**;
    - **CONVOLUTION** e **POOLING** layers tem como saida final as features de high-level da entrada;
    - As layers **fully-connected** usam essas features para classificar a imagem de entrada;
    - O resultado eh expresso como uma probabilidade dela ser de cada classe (softmax);
- Exemplo em codigo:

```
import tensorflow as tf

def generate_model():
    model = tf.keras.Sequential([
        # first conv layer
        # 32 filter maps
        # 3x3 filter size
        tf.keras.layers.Conv2D(32, filter_size=3, activation_function='relu')
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # second conv layer
        # downscaling our image, but increasing the amount of detected features
        # downsampling the irrelevant spatial information
        tf.keras.layers.Conv2D(64, filter_size=3, activation_function='relu')
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # fully connected classifier
        tf.keras.layers.Flatten()
        tf.kerras.layers.Dense(1024, activation='relu')
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model
```

## Detectacao de Objetos e Segmentacao de Imagens
- Dependendo da forma, detectar varios objetos e desenhar caixas ao redor deles na mesma imagem pode ser muito demorado, por isso existem varios algoritmos, mas um em especifico eh muito utilizado:
    - Faster R-CNN: Parte da rede treinada apenas para achar regioes que podem ter objetos das classes especificadas, depois disso, cada regiao passa por camadas convolucionais e, por fim, sao classificadas;
- Segmentacao:
    - Utilizamos uma FCN (fully convolutional network), em que a entrada eh uma imagem e a saida contem probabilidade para cada pixel;
    - FCN: Uma CNN em que a parte de classificacao eh substituida por outra CNN, entao se torna um "encoder", que eh o comeco da CNN e a ultima parte eh um "decoder", que retorna o vetor gerado por polling para uma imagem;
    - Essas layers sao analogas a uma CNN normal, pois fazem o oposto da mesma;
    - Code: `tf.keras.layers.Conv2DTranspose()`