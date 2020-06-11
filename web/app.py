import os
import cv2
import jsonpickle
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.models import model_from_json

app = Flask(__name__, template_folder = 'templates')
#UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

json_file = open('modelos/melhor_modelo.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights('modelos/melhor_peso.best.hdf5')


def teste():
    filestr = request.files['imagem'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img/255

    tam = 128
    img = cv2.resize(img, (tam, tam))

    teste = []
    teste.append(img)
    teste = np.array(teste)
    
    result = []
    pred = model.predict_on_batch(teste)
    result.append(pred)

    result = np.asarray(result)
    imprime = np.array(result[0][0])
    
    return np.argmax(imprime)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    aqui = teste()

    macacos = [ 'mantled howler',
                'patas monkey',
                'bald uakari',
                'japanese macaque',
                'pygmy marmoset',
                'white headed capuchin',
                'silvery marmoset',
                'common squirrel monkey',
                'black headed night monkey',
                'nilgiri langur']
    
    return render_template('index.html', text=str(macacos[aqui].upper()))




if __name__ == '__main__':
    app.run(debug=True)
