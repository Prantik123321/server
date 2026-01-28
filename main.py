import os
import io
import logging
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
import time

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

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize translator
translator = Translator()

# Language mapping for Tesseract
LANGUAGE_MAP = {
    'bn': 'ben',      # Bengali
    'en': 'eng',      # English
    'hi': 'hin',      # Hindi
    'es': 'spa',      # Spanish
    'fr': 'fra',      # French
    'de': 'deu',      # German
    'ja': 'jpn',      # Japanese
    'ko': 'kor',      # Korean
    'zh-cn': 'chi_sim',  # Chinese Simplified
    'ru': 'rus',      # Russian
    'ar': 'ara',      # Arabic
    'pt': 'por',      # Portuguese
}

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Preprocess image for better OCR accuracy"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        processed = cv2.medianBlur(processed, 3)
        
        # Convert back to PIL Image
        return Image.fromarray(processed)
    except Exception as e:
        logger.warning(f"Image preprocessing failed, using original: {e}")
        # Fallback to original image
        return Image.open(io.BytesIO(image_bytes))

def perform_ocr(image_bytes: bytes) -> Dict:
    """Perform OCR on image and return detected text and language"""
    try:
        start_time = time.time()
        
        # Preprocess image
        img = preprocess_image(image_bytes)
        
        # Try multiple language combinations for better accuracy
        lang_configs = [
            'eng+ben+hin',  # Primary languages
            'eng',
            'ben',
            'hin',
            'eng+ben',
            'eng+hin'
        ]
        
        best_text = ""
        best_lang = "unknown"
        
        for lang_config in lang_configs:
            try:
                config = f'--oem 3 --psm 6 -l {lang_config}'
                text = pytesseract.image_to_string(img, config=config)
                text = text.strip()
                
                if text and len(text) > len(best_text):
                    best_text = text
                    if 'ben' in lang_config:
                        best_lang = 'bn'
                    elif 'hin' in lang_config:
                        best_lang = 'hi'
                    else:
                        best_lang = 'en'
            except:
                continue
        
        if not best_text:
            # If no text detected with combinations, try auto
            text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
            best_text = text.strip()
        
        processing_time = time.time() - start_time
        
        return {
            "text": best_text,
            "language": best_lang,
            "processing_time": round(processing_time, 3)
        }
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

def translate_text(text: str, target_lang: str = "en") -> Dict:
    """Translate text to target language"""
    try:
        if not text.strip():
            return {
                "translated": "",
                "detected_lang": "unknown",
                "confidence": 0.0
            }
        
        # Detect language
        try:
            detected = translator.detect(text)
            detected_lang = detected.lang
            confidence = detected.confidence
        except:
            # Fallback detection
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
            "confidence": round(confidence, 2) if confidence else 0.5
        }
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        # Return original text if translation fails
        return {
            "translated": text,
            "detected_lang": "unknown",
            "confidence": 0.0
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "OCR Translation Server",
        "version": "1.0.0",
        "supported_languages": list(LANGUAGE_MAP.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    target_language: str = "en",
    ocr_only: bool = False
):
    """
    Process image: OCR + Translation
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 5MB)")
        
        # Perform OCR
        ocr_result = perform_ocr(contents)
        
        if ocr_only:
            return JSONResponse(content={
                "status": "success",
                "ocr": ocr_result,
                "translation": None
            })
        
        # Translate text
        if ocr_result["text"]:
            translation_result = translate_text(
                ocr_result["text"],
                target_language
            )
        else:
            translation_result = {
                "translated": "",
                "detected_lang": "unknown",
                "confidence": 0.0
            }
        
        return JSONResponse(content={
            "status": "success",
            "ocr": ocr_result,
            "translation": translation_result,
            "target_language": target_language,
            "timestamp": time.time()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/translate")
async def translate_only(text: str, target_language: str = "en"):
    """Translate text only (for testing)"""
    try:
        result = translate_text(text, target_language)
        return JSONResponse(content={
            "status": "success",
            "original": text,
            **result,
            "target_language": target_language
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )