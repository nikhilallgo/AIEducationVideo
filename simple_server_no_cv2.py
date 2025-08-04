"""
Simplified Llama Server - No OpenCV dependency
Handles text extraction and text-to-speech without complex dependencies
"""

import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
try:
    from PIL import Image
except ImportError:
    import PIL.Image as Image
import uvicorn
from typing import List, Dict, Any, Optional
import json
import logging
from contextlib import asynccontextmanager
import tempfile
from datetime import datetime
# If Tesseract isn't found automatically, add this to your server
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
llama_model = None
ocr_engine = None
ocr_type = None
tts_engine = None
tts_type = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all models when server starts"""
    global llama_model, ocr_engine, ocr_type, tts_engine, tts_type
    
    try:
        # Initialize Llama model for general AI tasks
        logger.info("Initializing Llama model...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            llama_model = {
                'tokenizer': tokenizer,
                'model': model,
                'type': 'transformers'
            }
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load transformers model: {e}")
            llama_model = {'type': 'fallback'}
        
        # Initialize OCR engine - simple approach
        logger.info("Initializing OCR engine...")
        ocr_engine = None
        ocr_type = None
        
        # Try Tesseract first (simpler, no NumPy conflicts)
        try:
            import pytesseract
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            ocr_engine = "tesseract"
            ocr_type = "Tesseract"
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.warning(f"Tesseract initialization failed: {e}")
        
        # Try EasyOCR if Tesseract failed
        if ocr_engine is None:
            try:
                import easyocr
                ocr_engine = easyocr.Reader(['en'], gpu=False)
                ocr_type = "EasyOCR"
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
        
        if ocr_engine is None:
            logger.warning("No OCR engine available - will provide test text")
        
        # Initialize TTS engine
        logger.info("Initializing TTS engine...")
        tts_engine = None
        tts_type = None
        
        # Try pyttsx3 first (offline, no network needed)
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 0.9)
            tts_type = "pyttsx3"
            logger.info("pyttsx3 TTS engine initialized successfully")
        except Exception as e:
            logger.warning(f"pyttsx3 initialization failed: {e}")
        
        # Try gTTS if pyttsx3 failed
        if tts_engine is None:
            try:
                from gtts import gTTS
                # Test gTTS with a simple phrase
                test_tts = gTTS(text="test", lang='en')
                tts_engine = "gtts"
                tts_type = "gTTS"
                logger.info("gTTS initialized successfully")
            except Exception as e:
                logger.warning(f"gTTS initialization failed: {e}")
        
        if tts_engine is None:
            logger.warning("No TTS engine available")
        
        logger.info(f"Initialization complete - OCR: {ocr_type}, TTS: {tts_type}")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
    
    yield  # Server runs here
    
    # Cleanup
    llama_model = None
    ocr_engine = None
    tts_engine = None

class SimpleLlamaProcessor:
    def __init__(self, llama_model, ocr_engine, ocr_type, tts_engine, tts_type):
        """Initialize processor with all models"""
        self.llama_model = llama_model
        self.ocr_engine = ocr_engine
        self.ocr_type = ocr_type
        self.tts_engine = tts_engine
        self.tts_type = tts_type
    
    async def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate text response using Llama model"""
        try:
            if self.llama_model['type'] == 'transformers':
                import torch
                tokenizer = self.llama_model['tokenizer']
                model = self.llama_model['model']
                
                inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                return response if response else "I understand your request."
            else:
                # Fallback responses
                if "detect" in prompt.lower():
                    return "I can see this appears to be an image with various content."
                elif "text" in prompt.lower():
                    return "I can help extract text from images when OCR is available."
                else:
                    return "I'm here to help you analyze images and extract information."
        except Exception as e:
            return f"I encountered an error: {str(e)}"
    
    def extract_text_from_image(self, pil_image) -> Dict[str, Any]:
        """Extract text from PIL image using available OCR"""
        try:
            logger.info(f"Starting text extraction using {self.ocr_type}")
            
            if self.ocr_engine is None:
                # Provide test text for audio verification
                test_text = "This is test text to verify the audio system is working properly. If you can hear this message, the text-to-speech functionality is operational."
                return {
                    'full_text': test_text,
                    'regions': [{'region_id': 0, 'text': test_text, 'confidence': 1.0}],
                    'region_count': 1,
                    'error': 'No OCR engine available',
                    'debug_info': 'Using test text for audio verification'
                }
            
            extracted_texts = []
            all_text = ""
            
            if self.ocr_type == "Tesseract":
                import pytesseract
                
                # Use PIL image directly with Tesseract
                text = pytesseract.image_to_string(pil_image)
                logger.info(f"Tesseract extracted text: '{text}'")
                
                if text.strip():
                    extracted_texts.append({
                        'region_id': 0,
                        'text': text.strip(),
                        'confidence': 0.8
                    })
                    all_text = text.strip()
                else:
                    logger.info("Tesseract found no text")
            
            elif self.ocr_type == "EasyOCR":
                # Convert PIL to numpy array for EasyOCR
                image_array = np.array(pil_image)
                results = self.ocr_engine.readtext(image_array)
                logger.info(f"EasyOCR found {len(results)} text regions")
                
                for i, (bbox, text, confidence) in enumerate(results):
                    if confidence >= 0.3:
                        extracted_texts.append({
                            'region_id': i,
                            'text': text,
                            'confidence': float(confidence)
                        })
                        all_text += text + " "
                        logger.info(f"EasyOCR text {i}: '{text}' (confidence: {confidence:.2f})")
            
            # If no text found, provide test text
            if not all_text.strip():
                logger.warning("No text extracted, providing test text")
                test_text = "No text was found in this image, but here is a test message to verify audio functionality is working correctly."
                extracted_texts.append({
                    'region_id': 0,
                    'text': test_text,
                    'confidence': 1.0
                })
                all_text = test_text
            
            result = {
                'full_text': all_text.strip(),
                'regions': extracted_texts,
                'region_count': len(extracted_texts),
                'ocr_engine': self.ocr_type,
                'debug_info': f"Used {self.ocr_type}, found {len(extracted_texts)} regions"
            }
            
            logger.info(f"Text extraction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            # Always return test text for audio verification
            test_text = "Text extraction encountered an error, but this test message will verify that audio functionality is working properly."
            return {
                'full_text': test_text,
                'regions': [{'region_id': 0, 'text': test_text, 'confidence': 1.0}],
                'region_count': 1,
                'error': str(e),
                'debug_info': f"OCR failed, using test text"
            }
    
    def detect_objects_simple(self, pil_image) -> Dict[str, Any]:
        """Simple object detection without OpenCV"""
        try:
            # Basic image analysis using PIL
            width, height = pil_image.size
            
            # Convert to grayscale for analysis
            gray = pil_image.convert('L')
            gray_array = np.array(gray)
            
            # Simple analysis
            mean_brightness = np.mean(gray_array)
            std_brightness = np.std(gray_array)
            
            objects = []
            
            # Simple heuristics
            if std_brightness < 30:  # Low variance suggests text/document
                objects.append({'class_name': 'document', 'confidence': 0.8})
            elif mean_brightness > 200:  # Bright image
                objects.append({'class_name': 'bright_image', 'confidence': 0.7})
            elif mean_brightness < 50:  # Dark image
                objects.append({'class_name': 'dark_image', 'confidence': 0.7})
            else:
                objects.append({'class_name': 'general_image', 'confidence': 0.6})
            
            # Add aspect ratio analysis
            aspect_ratio = width / height
            if aspect_ratio > 2:  # Wide image
                objects.append({'class_name': 'wide_content', 'confidence': 0.6})
            elif aspect_ratio < 0.5:  # Tall image
                objects.append({'class_name': 'tall_content', 'confidence': 0.6})
            
            return {
                'detections': objects,
                'count': len(objects)
            }
            
        except Exception as e:
            return {
                'detections': [{'class_name': 'unknown_object', 'confidence': 0.5}],
                'count': 1,
                'error': str(e)
            }
    
    def text_to_speech(self, text: str) -> str:
        """Convert text to speech and return audio file path"""
        try:
            logger.info(f"Converting text to speech using {self.tts_type}: '{text[:50]}...'")
            
            if not text.strip():
                logger.warning("Empty text provided for TTS")
                return None
            
            if self.tts_engine is None:
                logger.error("No TTS engine available")
                return None
            
            if self.tts_type == "pyttsx3":
                # Use pyttsx3
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                audio_path = temp_file.name
                temp_file.close()
                
                self.tts_engine.save_to_file(text, audio_path)
                self.tts_engine.runAndWait()
                
                logger.info(f"pyttsx3 audio saved to: {audio_path}")
                return audio_path
                
            elif self.tts_type == "gTTS":
                # Use gTTS
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                audio_path = temp_file.name
                temp_file.close()
                tts.save(audio_path)
                
                logger.info(f"gTTS audio saved to: {audio_path}")
                return audio_path
            
            return None
            
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return None

# Initialize FastAPI app
app = FastAPI(title="Simple Llama AI Assistant", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Simple Llama AI Assistant (No OpenCV)", 
        "status": "running",
        "capabilities": ["text_extraction", "simple_object_detection", "text_to_speech"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global llama_model, ocr_engine, ocr_type, tts_engine, tts_type
    return {
        "status": "healthy",
        "llama_loaded": llama_model is not None,
        "ocr_available": ocr_engine is not None,
        "ocr_type": ocr_type,
        "tts_available": tts_engine is not None,
        "tts_type": tts_type,
        "model_type": llama_model.get('type') if llama_model else None,
        "dependencies": "No OpenCV - using PIL only"
    }

@app.post("/analyze_image")
async def analyze_image_complete(file: UploadFile = File(...)):
    """Complete image analysis using PIL only"""
    global llama_model, ocr_engine, ocr_type, tts_engine, tts_type
    
    if llama_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        logger.info(f"Analyzing image: {file.filename}")
        
        # Read and decode image using PIL only
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        logger.info(f"Image loaded: {pil_image.size}, mode: {pil_image.mode}")
        
        # Create processor
        processor = SimpleLlamaProcessor(llama_model, ocr_engine, ocr_type, tts_engine, tts_type)
        
        # Extract text
        text_result = processor.extract_text_from_image(pil_image)
        logger.info(f"Text extraction complete: {text_result}")
        
        # Detect objects (simple analysis)
        object_result = processor.detect_objects_simple(pil_image)
        logger.info(f"Object detection complete: {object_result}")
        
        # Generate AI description
        description_prompt = f"Analyze this image: "
        if text_result['full_text']:
            description_prompt += f"Text found: '{text_result['full_text'][:100]}...' "
        if object_result['detections']:
            objects = [obj['class_name'] for obj in object_result['detections']]
            description_prompt += f"Objects detected: {', '.join(objects)}. "
        description_prompt += "Provide insights about this image."
        
        ai_description = await processor.generate_response(description_prompt)
        logger.info(f"AI description generated: {ai_description}")
        
        return JSONResponse(content={
            "filename": file.filename,
            "text_extraction": text_result,
            "object_detection": object_result,
            "ai_description": ai_description,
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/text_to_speech")
async def convert_text_to_speech(text: str = Form(...)):
    """Convert text to speech audio"""
    global llama_model, ocr_engine, ocr_type, tts_engine, tts_type
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        logger.info(f"TTS request for text: '{text[:50]}...'")
        
        processor = SimpleLlamaProcessor(llama_model, ocr_engine, ocr_type, tts_engine, tts_type)
        audio_path = processor.text_to_speech(text)
        
        if audio_path and os.path.exists(audio_path):
            logger.info(f"Returning audio file: {audio_path}")
            return FileResponse(
                audio_path,
                media_type="audio/wav",
                filename=f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
        else:
            logger.error("Audio generation failed")
            raise HTTPException(status_code=500, detail="Failed to generate audio - check server logs")
        
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8003))
    
    # Run server
    uvicorn.run(app, host=host, port=port)