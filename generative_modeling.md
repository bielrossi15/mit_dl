# Deep Generative Modeling

## Supervised x Unsupervised learning
- **Supervised learning:**
    - Dados (x,y): x sao os dados, y as labels;
    - Objetivo: aprender funcoes para mapear x->y;
    - Exemplos: classificacao, regressao, object detection, semantic segmentation, etc;
- **Unsupervised Learning:**
    - Dados x: x sao os dados, mas sem labels;
    - Objetivo: aprender a estrutura escondida ou subjacente dos dados;
    - Exemplos: clustering, feature/dimensionality reduction, etc;

## Generative modeling
- Objetivo: pegar uma entrada de treino de alguma distribuicao e aprender um modelo que representa a distribuicao;
- 2 jeitos para aprender:
    - Density estimation: Aprender a funcao de densidade probabilistica que descreve aonde o dado esta dentro dessa distribuicao;
    - Sample generation:
        - Training data ~Pdata(x);
        - Generated ~Pmodel(x);
        - Como aprender uma distribuicao probabilistica usando o modelo (Pmodel) que eh o mais parecido possivel com as entradas de treino? (Pdata);

## Modelo de Variavel Latente (nao manifesta, encoberta)
- Podemos aprender essas variaveis apenas observando dados?

# Autoencoders

## Background
- Um metodo para aprender representacoes de features de menor dimensao usando dados sem labels;
- "Encoder" aprende a mapear do dado, x, a um espaco de baixa dimensao latente, z;

  |   |
  |   |   | |
  | x |   | |   | z | 
  |   |   | | 
  |   |

- Porque nos preocupamos com esse z de baixa-dimensao?
    - Pois isso significa que podemos comprimir os dados em um vetor latente pequeno, aonde podemos aprender representacoes de features ricas e compactas;

## Como treinar?
- Treinar o modelo a usar essas features para reconstruir a entrada inicial;
- Um decoder;
- Decoder aprende a mapear inversamente do espaco latente z para uma reconstrucao observada, x_h;
- L(x, x_h) = ||x - x_h||**2 (mean squared error);

## Dimensoes do espaco latente
- Dimensao do espaco latente -> Qualidade de reconstrucao;
- Autoencoding eh uma forma de comprimir, entao quanto menor for o espaco latente, maior o gargalo do treino, menor a qualidade da reconstrucao;

## Autoencoders para aprendizado de representacao
- Hidden layer de gargalo: forca a rede neural a aprender uma representacao latente comprimida;
- Loss de reconstrucao: forca a representacao latente a capturar (encode) o maximo de "informacao" sobre os dados possivel;
- Autoencoding: Automatically encoding data;

# VAEs - Variational Autoencoders

## Diferencas
- Os autoencoders normais sao deterministicos, pois sua camada que gera o z (camada latente) eh apenas uma camada normal de uma rede neural, entao vai gerar apenas imagens iguais as que recebeu;
- VAEs geram versoes mais suaves das imagens de entrada e tambem criar imagens parecidas, nao so iguais;
- Troca aquela camada deterministica z com operacoes de amostragem estocastica:
    - Ao inves de aprender diretamente z, o VAE aprende uma media e uma variancia associada com a variavel latente que parametrizam uma distribuicao de probabilidade pra cada variavel latente;
- Entao temos um vetor de medias e um vetor de variancia;
- Para gerar novos dados retiramos amostras da distribuicao definida por esses novos vetores e gerar representacoes probabilisticas para esses espacos latentes;

## Otimizacao
- Encoder tenta aprender uma distribuicao probabilistica do espaco latente z dado os dados de entrada x;
- Decoder pega a representacao latente e computa uma nova distribuicao probabilistica;
- Encoder computa Qphi(z|x);
- Decoder computa Ptheta(x|z);
- Encoder e Decoder possuem diferentes sets de peso, phi e theta;
- Treinamos com uma funcao de custo: L(phi, theta, x) = (reconstruction loss) + (regularization term);
    - Reconstruction loss: parecido com o anterior, MSE, etc;
    - Regularization term: D(Qphi(z|x) || p(z));
        - p(z): previa do espaco latente z, uma hipotese inicial do que esperamos que a distribuicao de z sera;
        - D: termo de regularizacao, tenta fazer com o z que ela aprende a seguir essa distribuicao previa;
        - D: diminuir a divergencia entre o que o encoder tem como saida e a previa que colocamos;
        - Com isso, evitamos overfitting em certas partes do espaco latente, encorajando uma distribuicao parecida com a previa colocada;

## Previas (p(z))
- Escolha comum de previa - Gaussian normal;
    - p(z) = N(mu = 0; sigma**2 = 1), mu = media, sigma^2 = desvio padrao;
    - Encoraja o encoding a distribuir os encodings igualmente ao redor do centro do espaco latente;
    - Penaliza a rede quando tenta "roubar" ao clusterizar pontos em regioes especificas (ex: memorizar dados);
- -(1/2) * SUM(sigma_j + mu^2_j - 1 - log simga_j) -> KL-divergence;
    - Muito parecida com crossentropy loss, a distancia entre Qphi(z|x) e p(z);

## Regularizacao
- O que ganhamos por essa regularizacao?
- Continuidade: pontos que estao pertos no espaco latente sao similares apos serem decodificados;
    - Formas geometricas parecidas ficam perto etc;
    - Sem a regularizacao, sem tentar fazer com que a distribuicao probabilistica tome uma forma conhecida, a rede neural pode colocar pontos que representam formas parecidas distantes um do outro na distribuicao probabilistica resultante;

- Completude: amostragem do espaco latente tem um significado apos a decodificacao;
    - Sem a regularizacao, pontos que representam apenas desenhos aleatorios e insignificantes para a decodificacao podem acabar achando um espaco no espaco latente resultante;

- Sem regularizacao tambem temos o problema de que o modelo estaria apenas tentando otimizar para a diferenca da reconstruction loss, fazendo com que as variancias se tornariam quase que pontos de tao pequenas, e as medias poderiam ser completamente divergentes/diferentes, gerando descontinuidade (espacos sem nada);

- Com a regularizacao da media e da variancia, utilizando gaussian normal como previa, teriamos media = 0 e variancia = 1, entao as variaveis do espaco latente seriam obrigadas a tentar a ter a media centrada (zero) e todas as variancias tentariam ser regularizadas, garantindo suavidade;
- Centrar a media e regularizar as variancias;
- Quanto mais regularizamos, maior o risco de sofrer na qualidade do processo de reconstrucao e geracao;

## Backpropagation
- Nao podemos fazer a backpropagation de gradientes por camadas de amostragem, que possuem estocacidade;
- Backpropagation precisa de camadas deterministicas;
- VAEs tratam isso reparametrizando a operacao/camada de amostragem;
    - Consideram o vetor de amostragem latente z como uma soma de:
        - Um vetor de mu (media) fixo e,
        - Um vetor sigma (variancia) fixo, escalado por constantes randomicas retiradas da distribuicao previa;
        - z = mu + sigma . epsi;
        - Estocacidade eh o epsi;
        - epsi ~ N(0,1) (gaussian normal);
        - Ao inves de ter um no estocastico z ~ Qphi(z|x), temos um no deterministico z = g(phi, x, epsi), sendo epsi estocastico, mas considerado apenas como constantes;

## Perturbacao latente e Desemaranhamento
- Um efeito secundario e consequencia em impor essa previa na variavel latente eh que podemos pegar amostras dessas variaveis latentes e mexer nelas individualmente enquanto mantemos as outras variaveis fixas;
- Podemos rodar o decoder cada vez que essa variavel for "perturbada" e gerar diferentes saidas reconstruidas;
- EX: variacoes na pose da cabeca, sorriso, etc;
- Diferentes dimensoes de z encodam diferentes features latentes interpretaveis mantendo outras variaveis fixas e perturbando ela;
- Para otimizar VAEs e maximizar as informacoes que ela encoda, queremos que essas variaveis latentes sejam nao relacionadas umas com as outras, basicamente desenmaranhando-as;
- Com isso aprendemos a representacao mais rica e compacta possivel;

## Desenmaranhamento do espaco latente com Beta-VAEs
- L(theta,phi;x,z,beta) = Eqphi(z|x)[log ptheta(x|z)] (termo de reconstrucao) - Beta DKL (qphi(z|x) || p(z)) (termo de regularizacao);
- Beta controla a forca do termo de regularizacao, aplicando restricoes no encoding latente para encorajar o desemaranhamento;

     
