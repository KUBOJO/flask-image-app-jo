from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageFilter
import os
import cv2

# Path ke folder root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# Konfigurasi
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(input_path, output_path, effect):
    img = Image.open(input_path)

    if effect == 'grayscale':
        ImageOps.grayscale(img).save(output_path)
    elif effect == 'blur':
        img.filter(ImageFilter.GaussianBlur(5)).save(output_path)
    elif effect == 'rotate':
        img.rotate(90, expand=True).save(output_path)
    elif effect == 'mirror':
        ImageOps.mirror(img).save(output_path)
    elif effect == 'face_detect':
        img_cv = cv2.imread(input_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(output_path, img_cv)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file selected")

        file = request.files['image']
        effect = request.form.get('effect')

        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(original_path)

                processed_filename = f"processed_{filename}"
                processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                process_image(original_path, processed_path, effect)

                original_url = url_for('static', filename=f'uploads/{filename}')
                processed_url = url_for('static', filename=f'uploads/{processed_filename}')

                return render_template(
                    'index.html',
                    original=original_url,
                    processed=processed_url,
                    effect=effect
                )
            except Exception as e:
                return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
