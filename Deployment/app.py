from crypt import methods
from flask import Flask, flash, render_template, request, redirect
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import numpy as np
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

# directory for image upload
app.config['UPLOAD_FOLDER'] = 'uploads/images'

# checking file extension
ALLOWED_EXT = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

# loading our feature extracter (vgg16)
encoder = load_model()
def predict(image_path):
    if os.path.exists(image_path) == False:
        flash("File does'nt exists")
    img = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
    # expanding dimension of image
    img = img.reshape(-1, 224, 224, 3)
    # vgg16 preprocess
    img = preprocess_input(img)
    img_feat = encoder.predict(img)


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

        return render_template('home.html', filename=file_dir)
    else:
        flash('Allowed image extensions are jpg, jpeg, png')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)