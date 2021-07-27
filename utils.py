import spacy
import numpy as np

nlp_c = spacy.load('pt_core_news_sm', disable=['parser', 'ner', 'tagger', 'textcat'])

def tokenizador(texto):
  tokens_validos = []

  doc = nlp_c(texto)

  for token in doc:
    e_valido = (not token.is_stop) and token.is_alpha # verificação se o token não é stop word e é alphabetical
    if e_valido:
      tokens_validos.append((token.text).lower())

  return tokens_validos

def combinacao_vetores_soma(palavras, modelo):
  vetor_resultante = np.zeros(300)

  for pn in palavras:
    try:
      vetor_resultante += modelo.get_vector(pn)
    except KeyError:
      pass

  return vetor_resultante