import os
from crypt import methods
from flask import Flask, flash, render_template, request, redirect
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# received a warning for 10% allocation
# directory for image upload
app.config['UPLOAD_FOLDER'] = 'uploads/images'
MAX_CAP_LEN = 40

# checking file extension
ALLOWED_EXT = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

# loading our feature extracter (vgg16)

encoder = load_model(filepath='models/vgg16_model', compile=False)
decoder = load_model(filepath='models/model_30_sep', compile=False)
input_vocab = np.load('numpy/input_vocab.npy')

def generate_cap(img_feat):
    curr_word = tf.constant([[3]], dtype='int64')
    img_feat = tf.constant(img_feat)
    predicted = ''
    for i in range(MAX_CAP_LEN):
        # predict = decoder.predict(curr_word, img_feat)
        predict = decoder(curr_word, tf.constant(img_feat))
        word_predict = input_vocab[np.argmax(predict)]

        curr_word = tf.constant([[np.argmax(predict)]], dtype='int64')

        if word_predict == '<end>':
            return predicted
        predicted += word_predict + ' '
    return predicted
        
def predict(image_path):
    if os.path.exists(image_path) == False:
        flash("File does'nt exists")
    img = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
    # expanding dimension of image
    img = img.reshape(-1, 224, 224, 3)
    # vgg16 preprocess
    img = preprocess_input(img)
    img_feat = encoder.predict(img)
    
    cap = generate_cap(img_feat)

    return cap
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename == '':
        flash('No image selected!!')
        return redirect(request.url)
    elif file and allowed_file(file.filename):
        # saving file/image at 'uploads/images/img_name.jpg'
        file_dir = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_dir)    
        
        caption = predict(file_dir)
        # print(caption)
        os.remove(file_dir)
        return render_template('home.html', caption=caption)
    else:
        flash('Allowed image extensions are jpg, jpeg, png')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")