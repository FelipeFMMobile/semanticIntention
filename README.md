

# Similaridade Semântica & Análise de Intenção

## Escopo
Estudar modelos de ML que possam classificar a similaridade sêmantica desde uma análise básica com comparações de textos a partir de algorítimos de ML e de Rede Neurais com LSTM avançado. 

## Trabalhos 

- **Semantica com LaBSE**: Utiliza o embbeding LabSE (Agnóstico a linguagem) para comparar frases bases X frases alvo. Utiliza a similaridade do conseno.

- **PréProcessamento Lemmatizaion x Stemmatization**: Cria dois modelosde RNN LSTM: O primeiro pre-processa os embbedings usando **Stemmatização** e o segundo, usando **Lemmatização** usando o Pos Tagger MacMorpho para se adaptar as condiçõesda lingua PT.
Analise em database sintético básico e expandido. O resultado demonstra que a Lemmatização não se comportou bem com o modelo. 

* *Importante: Ambos apresentação overfitting por utilizar um database pequeno, mas com resultados bem distintos em um texto experimental.*

- **Sentence-Transformer (MiniLM) + Random Forest**: Gera embeddings com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilíngue) e classifica com Random Forest no notebook `semantic_distilBERT`. Não aplica stemmatização neste pipeline.

- **Stemming (inglês) + LSTM com Banking77**: Aplica tokenização em inglês com `SnowballStemmer("english")` e rede BiLSTM sobre o dataset público Banking77. Resultado atual salvo no notebook: acurácia de **0.7594** (macro F1 **0.75**, weighted F1 **0.76**).

### Similaridade LABSE
```
{'query': 'Cobra um valor', 'categoria': 'intenção de cobrança', 'texto_encontrado': 'Cobra um valor de 1 real e cinquenta centavos', 'score': 0.6940731406211853}

Score: 0.69

Notebook: semantic

```

### Análise Lemmatização
```
                texto classe_pred_lemm
0  Pagar conta de luz   Consulta Saldo
1       Trasferir R$5   Consulta Saldo
2   Informe meu saldo   Consulta Saldo
3     Cobra dez reais   Consulta Saldo

score: 0.99773  

Notebook: semantic_stem_X_lemm e semantic_stem_X_lemm_2

```

### Análise baseada em Stemmatização com Rede neural LSTM
```
                       texto     classe_pred  probabilidade  valor
0         Pagar conta de luz       Pagamento       0.871165    NaN
1              Trasferir R$5           Outro       0.511827   5.00
2          Informe meu saldo  Consulta Saldo       0.999577    NaN
3            Cobra dez reais        Cobrança       0.917386    NaN
4        me diga o meu saldo  Consulta Saldo       0.999577    NaN
5  passe um valor de R$50,00        Cobrança       0.999638  50.00
6          transfirir R$3,99           Outro       0.511827   3.99

score: 0.982108  

Notebook: semantic_stem_X_lemm e semantic_stem_X_lemm_2

```

### Análise baseada em Sentence-Transformer (MiniLM) + Random Forest

```
                       texto     classe_pred  probabilidade
0         pagar conta de luz           Outro            0.4
1              trasferir r$5       Pagamento            0.6
2          informe meu saldo  Consulta Saldo            0.5
3            cobra dez reais        Cobrança            0.5
4        me diga o meu saldo  Consulta Saldo            0.6
5  passe um valor de r$50,00        Cobrança            0.8
6          transfirir r$3,99        Cobrança            0.5
```

Observação: no split sintético o notebook reporta métricas muito altas (`precision/recall/f1` de 1.00), mas na amostra manual acima existem erros de classificação. Portanto, a leitura qualitativa desse experimento deve ser feita com cautela.

### Execução de referência adicional (amostra antiga do notebook `semantic_distilBERT`) 

```
0         Pagar conta de luz           Outro       0.993414
1              Trasferir R$5           Outro       0.999779
2          Informe meu saldo  Consulta Saldo       0.999255
3            Cobra dez reais           Outro       0.999999
4        me diga o meu saldo  Consulta Saldo       0.999255
5  passe um valor de R$50,00        Cobrança       0.999970
6          transfirir R$3,99           Outro       0.999779

Notebook: semantic_distilBERT

```

### Análise baseado em Stemming com Rede Neural LSTM - DataSet Banking 77

```
Sample (test):
                                                   text                   category                 label_pred      prob
964                         Where can the card be used?            card_acceptance            card_acceptance  0.346844
1966  Why is there more then one charge on my card...  transaction_charged_twice  transaction_charged_twice  0.892991
970      What are the rules to where I can use my...            card_acceptance           compromised_card  0.418855
1350                     Who else can top up my ac...         topping_up_by_card              verify_top_up  0.371930

Acurácia: 0.7594
Macro avg (P/R/F1): 0.77 / 0.75 / 0.75
Weighted avg (P/R/F1): 0.77 / 0.76 / 0.76

Notebook: semantic_lstm_banking77

```

## License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author 
Felipe Menezes, SÊnior iOS and Mobile Developer, Software Enginieer
[![Swift](https://img.shields.io/badge/Linkedin-profile-blue)](https://www.linkedin.com/in/felipe-menezes-dev)
