import base64
import io
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 현재 실행 중인 파일(main.py)의 디렉토리 절대 경로를 구합니다.
BASE_DIR = Path(__file__).resolve().parent

# 가중치 파일의 절대 경로 생성
MODEL_PATH = BASE_DIR / "models" / "net_params.pth"

# 시스템 경로에 프로젝트 루트를 추가하여, 어디서 실행하든 models 패키지를 찾을 수 있게 함
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# 모델 Import
try:
    # models/ccnet.py 파일 안의 ccnet 클래스(또는 함수) import
    from models.ccnet import ccnet
except ImportError as e:
    print(f"❌ [Error] 모델 모듈을 찾을 수 없습니다: {e}")
    print(f"👉 'models' 폴더 안에 '__init__.py'와 'ccnet.py'가 있는지 확인해주세요.")
    sys.exit(1)

# 설정 및 전역 변수 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None
transform = None

def make_tensor(x):
    """모델 출력이 튜플일 경우 첫 번째 요소(임베딩)만 추출"""
    if isinstance(x, tuple):
        x = x[0]
    return x

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 모델을 로드하고, 종료 시 정리하는 Lifespan 이벤트
    """
    global model, transform
    
    print(f"🚀 Starting server on {DEVICE}...")
    print(f"📂 Model path: {MODEL_PATH}")
    
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Weight file not found at: {MODEL_PATH}")

        model = ccnet(600, weight=0.8)
        
        
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        
 
        model.to(DEVICE)
        model.eval()
        

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"🔥 Critical Error loading model: {e}")
        raise e  

    yield
    print("👋 Shutting down...")

app = FastAPI(lifespan=lifespan)


class PalmRequest(BaseModel):
    base64_image: str


@app.post("/embed")
def get_embedding(request: PalmRequest): 
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
       
        try:
            image_data = base64.b64decode(request.base64_image)
        except Exception:
             raise ValueError("Invalid Base64 string")

        img = Image.open(io.BytesIO(image_data))
        

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            

        embedding_tensor = make_tensor(output)
      
        embedding_list = embedding_tensor.squeeze().cpu().numpy().tolist()
        
        return {
            "status": "success",
            "embedding_dim": len(embedding_list),
            "embedding": embedding_list
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Inference processing failed")

@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}