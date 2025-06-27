from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from werkzeug.utils import secure_filename

# إعداد Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.secret_key = 'super_secret_key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# تحميل نموذج الصور من Google Drive إذا لم يكن موجودًا
model_path = os.path.join('models', 'model.h5')
if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    gdown.download("https://drive.google.com/uc?id=1TE9y0vu_XfEZVxmwvizTsriKC6liAPxD", model_path, quiet=False)

model = load_model(model_path)
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# تحميل بيانات التدريب لتجهيز LabelEncoder
clinical_data = pd.read_csv("clinical_data_processed.csv")
le = LabelEncoder()
le.fit(clinical_data["Cancer_Type_Detailed"])

# ----------- توقع من صورة MRI -----------
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# ----------- توقع من ملفات سريرية CSV -----------
def predict_from_clinical_files(mrna_file, protein_file, mutation_file):
    predictions = []

    def process_file(file, model_path, label):
        if file:
            try:
                df = pd.read_csv(file).drop(columns=["Sample ID"], errors='ignore').fillna(0)
                if df.empty:
                    return
                model = joblib.load(model_path)
                preds = model.predict(df)
                predictions.extend(preds.tolist())
            except Exception as e:
                print(f"Error in {label} file: {e}")

    process_file(mrna_file, "models/model_mRNA.pkl", "mRNA")
    process_file(protein_file, "models/model_protein.pkl", "Protein")
    process_file(mutation_file, "models/model_mutation.pkl", "Mutation")

    if not predictions:
        return "❌ No valid data provided."

    final_pred = Counter(predictions).most_common(1)[0][0]
    final_label = le.inverse_transform([final_pred])[0]
    return f"Predicted Cancer Type: {final_label}"

# ----------- Routes -----------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            result, confidence = predict_tumor(path)
            session['mri_result'] = result
            session['confidence'] = f"{(confidence * 100 - 1.2):.2f}%"
            session['file_path'] = f'/uploads/{filename}'
            return redirect(url_for('index'))

    result = session.pop('mri_result', None)
    confidence = session.pop('confidence', None)
    file_path = session.pop('file_path', None)

    return render_template('index.html', result=result, confidence=confidence, file_path=file_path)

@app.route('/clinical', methods=['GET', 'POST'])
def clinical():
    if request.method == 'POST':
        mrna_file = request.files.get('mrna_file')
        protein_file = request.files.get('protein_file')
        mutation_file = request.files.get('mutation_file')
        result = predict_from_clinical_files(mrna_file, protein_file, mutation_file)
        session['clinical_result'] = result
        return redirect(url_for('clinical'))

    result = session.pop('clinical_result', None)
    return render_template('clinical.html', result=result)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

