import os
import io
import logging
import time
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import numpy as np
import cv2
from googletrans import Translator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCR Translation Server",
    description="Real-time OCR, language detection and translation API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize translator
translator = Translator()

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Preprocess image for better OCR accuracy"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(processed)
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return Image.open(io.BytesIO(image_bytes))

def perform_ocr(image_bytes: bytes) -> Dict:
    """Perform OCR on image"""
    try:
        start_time = time.time()
        
        # Preprocess image
        img = preprocess_image(image_bytes)
        
        # Try multiple language combinations
        lang_combinations = ['eng+ben+hin', 'eng', 'ben', 'hin', 'eng+ben', 'eng+hin']
        
        best_text = ""
        best_lang = "unknown"
        
        for lang_config in lang_combinations:
            try:
                config = f'--oem 3 --psm 6 -l {lang_config}'
                text = pytesseract.image_to_string(img, config=config).strip()
                
                if text and len(text) > len(best_text):
                    best_text = text
                    if 'ben' in lang_config and any(char in text for char in 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃেৈোৌ'):
                        best_lang = 'bn'
                    elif 'hin' in lang_config and any(char in text for char in 'अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूृेैोौ'):
                        best_lang = 'hi'
                    else:
                        best_lang = 'en'
            except:
                continue
        
        if not best_text:
            # Fallback to English only
            text = pytesseract.image_to_string(img, config='--oem 3 --psm 6').strip()
            best_text = text
        
        processing_time = time.time() - start_time
        
        return {
            "text": best_text,
            "language": best_lang,
            "processing_time": round(processing_time, 3),
            "success": bool(best_text)
        }
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {
            "text": "",
            "language": "unknown",
            "processing_time": 0,
            "success": False,
            "error": str(e)
        }

def translate_text(text: str, target_lang: str = "en") -> Dict:
    """Translate text to target language"""
    try:
        if not text.strip():
            return {
                "translated": "",
                "detected_lang": "unknown",
                "confidence": 0.0,
                "success": False
            }
        
        # Detect language
        try:
            detected = translator.detect(text)
            detected_lang = detected.lang
            confidence = detected.confidence or 0.0
        except:
            detected_lang = "en"
            confidence = 0.5
        
        # Translate if needed
        if detected_lang != target_lang and text.strip():
            translated = translator.translate(text, dest=target_lang, src=detected_lang)
            translated_text = translated.text
        else:
            translated_text = text
        
        return {
            "translated": translated_text,
            "detected_lang": detected_lang,
            "confidence": round(confidence, 2),
            "success": True
        }
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return {
            "translated": text,
            "detected_lang": "unknown",
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "OCR Translation Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process_image": "/api/process",
            "translate": "/api/translate"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "ocr-translation"
    }

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    target_language: str = "en"
):
    """
    Process image: OCR + Translation
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        
        # Size limit: 3MB
        if len(contents) > 3 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 3MB)")
        
        # Perform OCR
        ocr_result = perform_ocr(contents)
        
        if not ocr_result.get("success", False):
            return JSONResponse(content={
                "status": "error",
                "message": "No text detected in image",
                "ocr": ocr_result,
                "translation": None
            })
        
        # Translate text
        translation_result = translate_text(ocr_result["text"], target_language)
        
        return JSONResponse(content={
            "status": "success",
            "ocr": ocr_result,
            "translation": translation_result,
            "target_language": target_language,
            "timestamp": time.time()
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/translate")
async def translate_only(text: str, target_language: str = "en"):
    """Translate text only"""
    try:
        if not text.strip():
            return JSONResponse(content={
                "status": "error",
                "message": "No text provided"
            })
        
        result = translate_text(text, target_language)
        return JSONResponse(content={
            "status": "success" if result["success"] else "error",
            "original": text,
            **result,
            "target_language": target_language
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )