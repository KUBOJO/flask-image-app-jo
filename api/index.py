from flask import Flask, render_template, request, url_for
from PIL import Image, ImageOps, ImageFilter
from werkzeug.utils import secure_filename
import os
import cv2

# Lokasi dasar file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path template dan static (relatif terhadap file ini)
TEMPLATE_FOLDER = os.path.join(BASE_DIR, '../templates')
STATIC_FOLDER = os.path.join(BASE_DIR, '../static')

# Inisialisasi Flask
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)

# Konfigurasi folder upload
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder uploads jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Format gambar yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

# Fungsi deteksi wajah
def deteksi_wajah(input_path, output_path):
    # Gunakan path haarcascade lokal
    cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise Exception("❌ File haarcascade_frontalface_default.xml tidak ditemukan atau rusak.")

    img = cv2.imread(input_path)
    if img is None:
        raise Exception(f"❌ Gagal membaca gambar: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

# Routing utama
@app.route('/', methods=['GET', 'POST'])
def index():
    original_path = None
    processed_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        effect = request.form.get('effect')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_file_path)

            processed_filename = f"processed_{filename}"
            processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

            try:
                if effect in {'grayscale', 'blur', 'rotate', 'mirror'}:
                    img = Image.open(original_file_path)

                    if effect == 'grayscale':
                        processed_img = ImageOps.grayscale(img)
                    elif effect == 'blur':
                        processed_img = img.filter(ImageFilter.GaussianBlur(4))
                    elif effect == 'rotate':
                        processed_img = img.rotate(90)
                    elif effect == 'mirror':
                        processed_img = ImageOps.mirror(img)

                    processed_img.save(processed_file_path)

                elif effect == 'face_detect':
                    deteksi_wajah(original_file_path, processed_file_path)

                else:
                    return "❌ Efek tidak dikenali", 400

                # Konversi path untuk digunakan oleh HTML
                original_path = url_for('static', filename=f'uploads/{filename}')
                processed_path = url_for('static', filename=f'uploads/{processed_filename}')

            except Exception as e:
                print(f"[ERROR] {e}")
                return f"❌ Terjadi kesalahan saat memproses gambar: {e}", 500

        else:
            return "❌ File tidak valid atau tidak dipilih", 400

    return render_template('index.html', original_path=original_path, processed_path=processed_path)

# Jalankan server Flask (lokal)
if __name__ == "__main__":
    print("🚀 Flask berjalan di http://127.0.0.1:5050")
    app.run(debug=True, port=5050)
