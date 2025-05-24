from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import timm
import io

app = FastAPI(
    title="EfficientNet-Lite0 CIFAR-10 API",
    description="🚀 CPU-only image classifier using EfficientNet-Lite0.",
    version="1.0"
)

# Class names for CIFAR-10 (more can be included)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CPU only
device = torch.device("cpu")
torch.set_num_threads(4)
# torch.backends.quantized.engine = 'qnnpack'

# Model load
MODEL_PATH = r"C:\Users\Dell\Downloads\efficientnet_lite0_cifar10.pth"

def load_model():
    model = timm.create_model('efficientnet_lite0', pretrained=False, num_classes=10)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        top3_idx = probs.topk(3).indices.tolist()
        top3_scores = probs[top3_idx].tolist()
        predictions = {class_names[i]: float(top3_scores[j]) for j, i in enumerate(top3_idx)}

        return JSONResponse(content={"top3": predictions})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"message": "🚀 EfficientNet-Lite0 CIFAR-10 CPU API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

