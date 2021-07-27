from flask import Flask, make_response, render_template, flash
import pickle
from utils import tokenizador, combinacao_vetores_soma

from flask.globals import request
from gensim.models import KeyedVectors

app = Flask("Classificador_Artigos")
app.config['SECRET_KEY'] = 'secret key'

@app.before_first_request
def before_first_request():
    global w2v_model
    global lr

    w2v_model = KeyedVectors.load_word2vec_format('models\modelo_skipgram.txt')
    
    with open('models\lr_skip.pkl', 'rb') as f:
        lr = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def inicio():
    if request.method == 'POST':
        title = request.form.get('titulo')
        print(title)

        palavras = tokenizador(title)
        print(palavras)

        vetor = combinacao_vetores_soma(palavras, w2v_model)
        vetor = vetor.reshape(1, -1)

        categoria = lr.predict(vetor)
        print(categoria)

        out = categoria[0].capitalize()

        return render_template('artigo.html', titulo=title, categoria=out)
    return render_template('artigo.html')


if __name__ == "__main__":
    app.run(debug=True)