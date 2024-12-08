from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
import onnxruntime as ort

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Daftar URL frontend yang diizinkan
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode HTTP
    allow_headers=["*"],  # Izinkan semua header
)


# Load ONNX model
onnx_model_path = "blood_cells_v8.onnx"
session = ort.InferenceSession(onnx_model_path)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}



# Preprocessing function sesuai kebutuhan model ONNX
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize sesuai kebutuhan model
    image_array = np.array(image).astype("float32") / 255.0  # Normalisasi 0-1
    if len(image_array.shape) == 2:  # Jika grayscale, tambahkan channel
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.transpose(image_array, (2, 0, 1))  # ONNX butuh format CHW
    return np.expand_dims(image_array, axis=0)  # Tambahkan batch size = 1

@app.post("/get-prediction")
async def get_prediction(file: UploadFile = File(...)):
    try:
        # Baca file gambar
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess gambar
        input_tensor = preprocess_image(image)
        
        # Inferensi dengan ONNX Runtime
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        # Ambil kelas dengan skor tertinggi dan confidence
        class_probabilities = outputs[0][0]  # Ambil probabilitas kelas
        max_confidence = np.max(class_probabilities)
        class_id = np.argmax(class_probabilities)
        
        # Daftar nama kelas
        class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']  # Sesuaikan dengan dataset Anda
        
        # Tentukan threshold confidence (misalnya 0.5 atau 50%)
        CONFIDENCE_THRESHOLD = 0.4
        
        if max_confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Undetected"
        else:
            predicted_label = class_names[class_id] if class_id < len(class_names) else "Unknown"
        
        return JSONResponse(content={
            "predicted_class": predicted_label,
            "confidence": float(max_confidence)
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
