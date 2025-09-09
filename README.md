

# Similaridade Semântica & Análise de Intenção

## Escopo
Estudar modelos de ML que possam classificar a similaridade sêmantica desde uma análise básica com comparações de textos a partir de algorítimos de ML e de Rede Neurais com LSTM avançado. 

## Trabalhos 

- **Semantica com LaBSE**: Utiliza o embbeding LabSE (Agnóstico a linguagem) para comparar frases bases X frases alvo. Utiliza a similaridade do conseno.

- **PréProcessamento Lemmatizaion x Stemmatization**: Cria dois modelosde RNN LSTM: O primeiro pre-processa os embbedings usando **Stemmatização** e o segundo, usando **Lemmatização** usando o Pos Tagger MacMorpho para se adaptar as condiçõesda lingua PT.
Analise em database sintético básico e expandido. O resultado demonstra que a Lemmatização não se comportou bem com o modelo. 

* *Importante: Ambos apresentação overfitting por utilizar um database pequeno, mas com resultados bem distintos em um texto experimental.*

- **Stemmatization com DistilBERT**: Aplica o pré processamento com base no modelo DistilBERT para embbeding dos textos, aplica o algoritimo de classificação Random Forest. Apesar de um embbeding grande, apresenta resultados inferiores a rede neural.

- **Stemmatization LSTM com Banking77**: Aplica o melhor resultado encontrando usando Stemmatização sobre um dataset público Banking77 com a mesma rede LSTM para analisar o score. Atinge uma acurácia de 0.75 

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

### Análise baseada em DistilBERT

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


### Análise baseado em Stemming com Rede Neural LSTM Robusta (REVER) 

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
                                                   text  \
182                I have a 1 euro fee on my statement.   
1054  I have paid money into my account but it doesn...   
2975             how do i add money with my apple watch   
2457    I'm looking for the option to top up by cheque.   
1782  Did my refund go through? It's not on my state...   
1014  There appears to have been a reversion in my t...   
767   The ATM gave me the wrong amount of cash today...   
256   What fiat currencies are supported for holding...   
83        Where do you guys acquire your exchange rate?   
1588                              Why can't I get cash?   

                                              category  \
182                          extra_charge_on_statement   
1054  balance_not_updated_after_cheque_or_cash_deposit   
2975                           apple_pay_or_google_pay   
2457                          top_up_by_cash_or_cheque   
1782                             Refund_not_showing_up   
1014                                   top_up_reverted   
767                      wrong_amount_of_cash_received   
256                              fiat_currency_support   
83                                       exchange_rate   
1588                          declined_cash_withdrawal   

                              label_pred      prob  
182            extra_charge_on_statement  0.490520  o
1054  transfer_not_received_by_recipient  0.305712  X
2975             apple_pay_or_google_pay  0.962922  o
2457            top_up_by_cash_or_cheque  0.619489  o
1782               Refund_not_showing_up  0.906209  o
1014                      pending_top_up  0.239173  X
767       cash_withdrawal_not_recognised  0.781609  X
256                fiat_currency_support  0.991938  o
83                         exchange_rate  0.932884  o
1588            declined_cash_withdrawal  0.410309  o

score: 0.7593

Notebook: semantic_lstm_banking77

```

## License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author 
Felipe Menezes, SÊnior iOS and Mobile Developer, Software Enginieer
[![Swift](https://img.shields.io/badge/Linkedin-profile-blue)](https://www.linkedin.com/in/felipe-menezes-dev)

