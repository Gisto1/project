from flask import Flask, render_template, redirect, url_for
from flask_restful import Resource, Api, reqparse
from flask_wtf import FlaskForm
import werkzeug, os
from flask_uploads import UploadSet, IMAGES, configure_uploads
from keras.models import load_model
from werkzeug.utils import secure_filename

import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
api = Api(app)
app.secret_key = "you-will-not-get-this"
UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

from wtforms import StringField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired

locust_classifier = load_model('locusts.h5')
acanth_classifier = load_model('acanth.h5')

class UploadForm(FlaskForm):
    image = FileField('Recipe Image', validators=[FileRequired(), FileAllowed(['jpg','png','JPG'], 'Images only!')])


@app.route('/', methods=["GET","POST"])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.image.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('result', filename=filename))
    return render_template('/index.html', form=form)

@app.route('/result/<filename>')
def result(filename):
    locust_classifier = load_model('locusts.h5')
    acanth_classifier = load_model('acanth.h5')
    test_image = image.load_img('static/img/'+filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = acanth_classifier.predict(test_image)
    if result[0][0] == 0:
        prediction = 'Acanthoplus'
    elif locust_classifier.predict(test_image) == 1:
        prediction = 'Locust'
    else:
        prediction = 'Image Not of corncen'
    return render_template('success.html', prediction=prediction)


class PhotoUpload(Resource): 
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        photo = data['file']

        print(data)

        if photo:
            filename = 'your_image.png'
            photo.save(os.path.join(UPLOAD_FOLDER,filename))
            return {
                    'data':'',
                    'message':'photo uploaded',
                    'status':'success'
                    }
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


api.add_resource(PhotoUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=False)
