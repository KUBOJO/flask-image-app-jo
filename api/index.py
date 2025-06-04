from flask import Flask, render_template, request, url_for
from PIL import Image, ImageOps, ImageFilter
from werkzeug.utils import secure_filename
import os
import cv2

# Lokasi dasar file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path template dan static (relatif terhadap file ini) - DIPERBAIKI
TEMPLATE_FOLDER = os.path.join(BASE_DIR, '../../templates')  # Naik 2 level ke root
STATIC_FOLDER = os.path.join(BASE_DIR, '../../static')      # Naik 2 level ke root

# Inisialisasi Flask
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)

# Konfigurasi folder upload
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder uploads jika belum ada - DIPERBAIKI (handle error)
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    print(f"❌ Gagal membuat folder upload: {e}")

# Format gambar yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

# Fungsi deteksi wajah - DIPERBAIKI (error handling)
def deteksi_wajah(input_path, output_path):
    try:
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        
        # Cek apakah file HaarCascade ada
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("File haarcascade tidak ditemukan di path: " + cascade_path)
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            raise Exception("❌ File haarcascade rusak atau tidak terbaca")

        img = cv2.imread(input_path)
        if img is None:
            raise Exception(f"❌ Gambar tidak terbaca: {input_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not cv2.imwrite(output_path, img):
            raise Exception("❌ Gagal menyimpan gambar hasil deteksi")
            
    except Exception as e:
        print(f"[DETEKSI ERROR] {str(e)}")
        raise  # Re-raise error untuk ditangkap di route utama

# Routing utama - DIPERBAIKI (log lebih informatif)
@app.route('/', methods=['GET', 'POST'])
def index():
    original_path = None
    processed_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        effect = request.form.get('effect')

        if not file:
            return "❌ Tidak ada file yang dipilih", 400
            
        if not allowed_file(file.filename):
            return "❌ Format file tidak didukung (hanya .png, .jpg, .jpeg)", 400

        try:
            filename = secure_filename(file.filename)
            original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Simpan file upload
            file.save(original_file_path)
            print(f"✅ File disimpan: {original_file_path}")

            processed_filename = f"processed_{filename}"
            processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

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
                print(f"✅ Efek {effect} berhasil diproses")

            elif effect == 'face_detect':
                deteksi_wajah(original_file_path, processed_file_path)
                print("✅ Deteksi wajah berhasil")
                
            else:
                return "❌ Efek tidak valid", 400

            # Generate URL untuk template
            original_path = url_for('static', filename=f'uploads/{filename}')
            processed_path = url_for('static', filename=f'uploads/{processed_filename}')

        except Exception as e:
            print(f"[ROUTE ERROR] {str(e)}")
            return f"❌ Gagal memproses: {str(e)}", 500

    return render_template('index.html', original_path=original_path, processed_path=processed_path)

# Jalankan server Flask (lokal)
if __name__ == "__main__":
    print("\n🚀 Server Flask siap di http://localhost:5050")
    print(f"🔧 Template folder: {TEMPLATE_FOLDER}")
    print(f"📁 Static folder: {STATIC_FOLDER}\n")
    app.run(debug=True, port=5050)