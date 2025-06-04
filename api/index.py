from flask import Flask, render_template, request, url_for
from PIL import Image, ImageOps, ImageFilter
from werkzeug.utils import secure_filename
import os
import cv2

# Lokasi dasar file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path template dan static (relatif terhadap file ini)
TEMPLATE_FOLDER = os.path.join(BASE_DIR, '../../templates')  # Naik 2 level ke root
STATIC_FOLDER = os.path.join(BASE_DIR, '../../static')      # Naik 2 level ke root

# Inisialisasi Flask
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)

# Konfigurasi folder upload
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Batas upload 5MB

# Buat folder uploads jika belum ada
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    print(f"❌ Gagal membuat folder upload: {e}")

# Format gambar yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

# Fungsi deteksi wajah dengan optimasi
def deteksi_wajah(input_path, output_path):
    try:
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"File haarcascade tidak ditemukan di: {cascade_path}")
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            raise Exception("❌ Gagal memuat file haarcascade")

        img = cv2.imread(input_path)
        if img is None:
            raise Exception(f"❌ Tidak dapat membaca gambar: {input_path}")

        # Konversi ke grayscale untuk deteksi lebih cepat
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah dengan parameter optimal
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Gambar kotak di sekitar wajah
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not cv2.imwrite(output_path, img):
            raise Exception("❌ Gagal menyimpan hasil deteksi")
            
    except Exception as e:
        print(f"[ERROR DETEKSI] {str(e)}")
        raise

# Routing utama dengan error handling
@app.route('/', methods=['GET', 'POST'])
def index():
    original_path = None
    processed_path = None

    if request.method == 'POST':
        # Validasi file upload
        if 'image' not in request.files:
            return render_template('error.html', message="❌ Tidak ada file yang dipilih"), 400
            
        file = request.files['image']
        effect = request.form.get('effect', '')

        if file.filename == '':
            return render_template('error.html', message="❌ Nama file kosong"), 400
            
        if not (file and allowed_file(file.filename)):
            return render_template('error.html', message="❌ Format file tidak didukung"), 400

        try:
            # Proses penyimpanan file
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")
            
            file.save(original_path)
            print(f"📁 File disimpan: {original_path}")

            # Proses efek gambar
            if effect == 'grayscale':
                img = Image.open(original_path)
                ImageOps.grayscale(img).save(processed_path)
            elif effect == 'blur':
                img = Image.open(original_path)
                img.filter(ImageFilter.GaussianBlur(5)).save(processed_path)
            elif effect == 'rotate':
                img = Image.open(original_path)
                img.rotate(90, expand=True).save(processed_path)
            elif effect == 'mirror':
                img = Image.open(original_path)
                ImageOps.mirror(img).save(processed_path)
            elif effect == 'face_detect':
                deteksi_wajah(original_path, processed_path)
            else:
                return render_template('error.html', message="❌ Efek tidak valid"), 400

            # Generate URL untuk ditampilkan
            original_url = url_for('static', filename=f'uploads/{filename}')
            processed_url = url_for('static', filename=f'uploads/processed_{filename}')
            
            return render_template('result.html', 
                               original=original_url, 
                               processed=processed_url,
                               effect=effect)

        except Exception as e:
            print(f"🔥 Error: {str(e)}")
            return render_template('error.html', message=f"❌ Gagal memproses: {str(e)}"), 500

    return render_template('upload.html')

# Konfigurasi untuk production
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Port default Render
    app.run(host='0.0.0.0', port=port, debug=False)