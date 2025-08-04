import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import fitz  # PyMuPDF
import onnxruntime as ort
import pyttsx3
from PIL import Image, ImageTk
import tempfile
import json
import logging

# Optional imports for enhanced TTS
try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("gTTS not available. Install with: pip install gtts pygame")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Optional imports for translation
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("Google Translate not available. Install with: pip install googletrans==4.0.0-rc1")

try:
    from indic_transliteration import sanscript
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    print("Indic transliteration not available. Install with: pip install indic-transliteration")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgePDFProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Edge PDF AI Processor")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.current_pdf_path = None
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.extracted_text = ""
        self.detected_objects = []
        self.tts_playing = False
        self.current_image_with_boxes = None
        self.show_bounding_boxes = tk.BooleanVar(value=True)
        self.tts_method = None  # Will be set in GUI setup
        
        # Translation variables
        self.translator = None
        self.original_text = ""
        self.translated_text = ""
        self.current_display_language = "original"
        
        # AI Models
        self.object_detection_session = None
        self.tts_engine = None
        
        # Setup GUI
        self.setup_gui()
        self.setup_ai_models()
        
    def setup_gui(self):
        """Setup the main GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Edge PDF AI Processor", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="PDF Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Select PDF", command=self.select_pdf).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Left panel - PDF viewer
        left_panel = ttk.LabelFrame(content_frame, text="PDF Viewer", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        
        # PDF navigation
        nav_frame = ttk.Frame(left_panel)
        nav_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        nav_frame.columnconfigure(1, weight=1)
        
        ttk.Button(nav_frame, text="◀", command=self.prev_page).grid(row=0, column=0)
        self.page_label = ttk.Label(nav_frame, text="Page: 0/0")
        self.page_label.grid(row=0, column=1)
        ttk.Button(nav_frame, text="▶", command=self.next_page).grid(row=0, column=2)
        
        # Bounding box toggle
        bbox_frame = ttk.Frame(left_panel)
        bbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.bbox_checkbox = ttk.Checkbutton(
            bbox_frame, 
            text="Show Bounding Boxes", 
            variable=self.show_bounding_boxes,
            command=self.toggle_bounding_boxes
        )
        self.bbox_checkbox.grid(row=0, column=0)
        
        ttk.Button(bbox_frame, text="Clear Boxes", command=self.clear_bounding_boxes).grid(row=0, column=1, padx=(10, 0))
        
        # PDF display canvas
        self.pdf_canvas = tk.Canvas(left_panel, bg="white", width=400, height=500)
        self.pdf_canvas.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel - Processing results
        right_panel = ttk.LabelFrame(content_frame, text="AI Processing Results", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        right_panel.rowconfigure(3, weight=1)
        
        # Processing buttons
        button_frame = ttk.Frame(right_panel)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        ttk.Button(button_frame, text="Extract Text", command=self.extract_text).grid(row=0, column=0, padx=(0, 2), sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Detect Objects", command=self.detect_objects).grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Clear Detection", command=self.clear_detection).grid(row=0, column=2, padx=(2, 0), sticky=(tk.W, tk.E))
        
        # Translation controls
        translation_frame = ttk.LabelFrame(right_panel, text="Translation")
        translation_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        translation_frame.columnconfigure(0, weight=1)
        translation_frame.columnconfigure(1, weight=1)
        translation_frame.columnconfigure(2, weight=1)
        
        ttk.Button(translation_frame, text="Translate to Kannada", command=self.translate_to_kannada).grid(row=0, column=0, padx=(0, 2), sticky=(tk.W, tk.E))
        ttk.Button(translation_frame, text="Show Original", command=self.show_original_text).grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(translation_frame, text="Show Kannada", command=self.show_kannada_text).grid(row=0, column=2, padx=(2, 0), sticky=(tk.W, tk.E))
        
        # Extracted text display
        text_frame = ttk.LabelFrame(right_panel, text="Extracted Text")
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
        # Text language indicator
        self.text_language_label = ttk.Label(text_frame, text="Language: Original", font=("Arial", 9, "italic"))
        self.text_language_label.grid(row=0, column=0, sticky=(tk.W), padx=5, pady=(5, 0))
        
        self.text_display = scrolledtext.ScrolledText(text_frame, height=8, wrap=tk.WORD, font=("Arial", 10))
        self.text_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Audio controls
        audio_frame = ttk.LabelFrame(right_panel, text="Text-to-Speech")
        audio_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_frame.columnconfigure(0, weight=1)
        audio_frame.columnconfigure(1, weight=1)
        audio_frame.columnconfigure(2, weight=1)
        
        # TTS method selection
        tts_method_frame = ttk.Frame(audio_frame)
        tts_method_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(tts_method_frame, text="TTS Method:").grid(row=0, column=0, padx=(0, 5))
        self.tts_method = tk.StringVar(value="system")
        tts_combo = ttk.Combobox(tts_method_frame, textvariable=self.tts_method, width=15)
        tts_combo['values'] = []
        if GTTS_AVAILABLE:
            tts_combo['values'] = ('system', 'google')
        else:
            tts_combo['values'] = ('system',)
        tts_combo.grid(row=0, column=1)
        tts_combo.state(['readonly'])
        
        # Language/Voice buttons
        ttk.Button(audio_frame, text="Play English", command=lambda: self.play_tts("en")).grid(row=1, column=0, padx=(0, 2), sticky=(tk.W, tk.E))
        ttk.Button(audio_frame, text="Play Kannada", command=lambda: self.play_tts("kn")).grid(row=1, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(audio_frame, text="Stop Audio", command=self.stop_tts).grid(row=1, column=2, padx=(2, 0), sticky=(tk.W, tk.E))
        
        # Object detection results
        objects_frame = ttk.LabelFrame(right_panel, text="Detected Objects")
        objects_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        objects_frame.columnconfigure(0, weight=1)
        objects_frame.rowconfigure(0, weight=1)
        
        self.objects_display = scrolledtext.ScrolledText(objects_frame, height=6, wrap=tk.WORD)
        self.objects_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select a PDF to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def setup_ai_models(self):
        """Initialize AI models for edge processing"""
        try:
            self.status_var.set("Initializing AI models...")
            
            # Setup ONNX Runtime with Qualcomm HTP backend
            self.setup_onnx_runtime()
            
            # Setup TTS engine
            self.setup_tts_engine()
            
            # Setup translation engine
            self.setup_translation_engine()
            
            self.status_var.set("AI models initialized successfully")
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing AI models: {str(e)}"
            self.status_var.set(error_msg)
            logger.error(error_msg)
            messagebox.showerror("Model Initialization Error", error_msg)
    
    def setup_onnx_runtime(self):
        """Setup ONNX Runtime with Qualcomm HTP backend"""
        try:
            # Check for available providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # Try to use QNN (Qualcomm Neural Network) provider if available
            providers = []
            if 'QNNExecutionProvider' in available_providers:
                providers.append(('QNNExecutionProvider', {
                    'backend_path': 'QnnHtp.so',  # Qualcomm HTP backend
                    'profiling_level': 'basic'
                }))
                logger.info("Using Qualcomm HTP backend")
            elif 'CPUExecutionProvider' in available_providers:
                providers.append('CPUExecutionProvider')
                logger.warning("Qualcomm HTP not available, falling back to CPU")
            
            # Load a lightweight object detection model (you would replace this with your actual model)
            # For demonstration, we'll create a placeholder
            self.object_detection_providers = providers
            logger.info(f"ONNX Runtime configured with providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
            
        except Exception as e:
            logger.error(f"Error setting up ONNX Runtime: {str(e)}")
            raise
    
    def setup_tts_engine(self):
        """Setup text-to-speech engine"""
        try:
            # Initialize system TTS
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            # Try to find English and Kannada voices
            self.english_voice = None
            self.kannada_voice = None
            
            logger.info("Available System TTS voices:")
            for i, voice in enumerate(voices):
                logger.info(f"Voice {i}: {voice.id} - {voice.name} - Languages: {getattr(voice, 'languages', 'Unknown')}")
                
                # Check for English voices
                if any(lang in voice.id.lower() for lang in ['en', 'english']):
                    self.english_voice = voice.id
                elif any(lang in voice.name.lower() for lang in ['english', 'david', 'zira', 'mark']):
                    self.english_voice = voice.id
                
                # Check for Kannada voices (rarely available on Windows)
                if any(lang in voice.id.lower() for lang in ['kn', 'kannada', 'kan']):
                    self.kannada_voice = voice.id
                elif any(lang in voice.name.lower() for lang in ['kannada', 'kan']):
                    self.kannada_voice = voice.id
            
            # Default to first available voice if specific ones not found
            if not self.english_voice and voices:
                self.english_voice = voices[0].id
            
            # Check Google TTS availability
            if GTTS_AVAILABLE:
                try:
                    # Initialize pygame for audio playback
                    pygame.mixer.init()
                    logger.info("Google TTS (gTTS) is available and will be used for Kannada")
                    self.gtts_available = True
                except Exception as e:
                    logger.warning(f"Pygame initialization failed: {str(e)}")
                    self.gtts_available = False
            else:
                logger.warning("Google TTS not available. Install with: pip install gtts pygame")
                self.gtts_available = False
            
            logger.info(f"Selected English voice: {self.english_voice}")
            logger.info(f"Kannada TTS method: {'Google TTS' if self.gtts_available else 'System (English fallback)'}")
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up TTS engine: {str(e)}")
            raise
    
    def setup_translation_engine(self):
        """Setup translation services"""
        try:
            if GOOGLETRANS_AVAILABLE:
                self.translator = Translator()
                logger.info("Google Translator initialized successfully")
            else:
                logger.warning("Google Translate not available. Install with: pip install googletrans==4.0.0-rc1")
                self.translator = None
            
            if TRANSLITERATION_AVAILABLE:
                logger.info("Indic transliteration available")
            else:
                logger.warning("Indic transliteration not available. Install with: pip install indic-transliteration")
            
        except Exception as e:
            logger.error(f"Error setting up translation engine: {str(e)}")
            self.translator = None
    
    def check_internet_connection(self):
        """Check if internet connection is available"""
        try:
            if not REQUESTS_AVAILABLE:
                return False
            response = requests.get("https://www.google.com", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def select_pdf(self):
        """Open file dialog to select PDF"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_pdf(file_path)
    
    def load_pdf(self, file_path):
        """Load and display PDF"""
        try:
            self.current_pdf_path = file_path
            self.pdf_document = fitz.open(file_path)
            self.total_pages = len(self.pdf_document)
            self.current_page = 0
            
            # Update UI
            self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.update_page_display()
            
            self.status_var.set(f"PDF loaded successfully - {self.total_pages} pages")
            logger.info(f"PDF loaded: {file_path} ({self.total_pages} pages)")
            
        except Exception as e:
            error_msg = f"Error loading PDF: {str(e)}"
            self.status_var.set(error_msg)
            logger.error(error_msg)
            messagebox.showerror("PDF Loading Error", error_msg)
    
    def update_page_display(self):
        """Update the PDF page display"""
        if not self.pdf_document:
            return
        
        try:
            # Get current page
            page = self.pdf_document[self.current_page]
            
            # Render page as image
            mat = fitz.Matrix(1.5, 1.5)  # Zoom factor
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Store original image for bounding box drawing
            self.original_pil_image = pil_image.copy()
            
            # Draw bounding boxes if enabled and objects exist
            if self.show_bounding_boxes.get() and self.detected_objects:
                pil_image = self.draw_bounding_boxes_on_image(pil_image)
            
            # Resize to fit canvas
            canvas_width = self.pdf_canvas.winfo_width() or 400
            canvas_height = self.pdf_canvas.winfo_height() or 500
            pil_image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.pdf_canvas.delete("all")
            self.pdf_canvas.create_image(canvas_width//2, canvas_height//2, 
                                       image=self.current_image, anchor=tk.CENTER)
            
            # Update page label
            self.page_label.config(text=f"Page: {self.current_page + 1}/{self.total_pages}")
            
        except Exception as e:
            logger.error(f"Error updating page display: {str(e)}")
    
    def draw_bounding_boxes_on_image(self, pil_image):
        """Draw bounding boxes on PIL image"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create a copy to draw on
            img_with_boxes = pil_image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # Get image dimensions for scaling
            img_width, img_height = pil_image.size
            
            # Define colors for different object types
            colors = {
                'text_region': '#FF0000',      # Red
                'table_line': '#00FF00',       # Green
                'figure_or_image': '#0000FF',  # Blue
                'default': '#FFFF00'           # Yellow
            }
            
            # Draw each bounding box
            for obj in self.detected_objects:
                bbox = obj['bbox']
                obj_class = obj.get('class', 'default')
                confidence = obj.get('confidence', 0.0)
                
                # Scale bounding box coordinates to image size
                # Note: detected objects coordinates are from detection image,
                # we need to scale them to display image
                x, y, w, h = bbox
                
                # Get color for this object type
                color = colors.get(obj_class, colors['default'])
                
                # Draw rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                
                # Draw label with confidence
                label = f"{obj_class}: {confidence:.2f}"
                
                # Try to use a font, fallback to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                # Calculate text size for background
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Draw background for text
                draw.rectangle([x, y - text_height - 2, x + text_width + 4, y], 
                              fill=color, outline=color)
                
                # Draw text
                draw.text((x + 2, y - text_height - 1), label, fill='white', font=font)
            
            return img_with_boxes
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {str(e)}")
            return pil_image
    
    def toggle_bounding_boxes(self):
        """Toggle bounding box display"""
        self.update_page_display()
    
    def clear_bounding_boxes(self):
        """Clear all bounding boxes and detected objects"""
        self.detected_objects = []
        self.objects_display.delete(1.0, tk.END)
        self.objects_display.insert(tk.END, "No objects detected")
        self.update_page_display()
        self.status_var.set("Bounding boxes cleared")
    
    def clear_detection(self):
        """Clear detection results"""
        self.clear_bounding_boxes()
    
    def prev_page(self):
        """Navigate to previous page"""
        if self.pdf_document and self.current_page > 0:
            self.current_page -= 1
            self.update_page_display()
    
    def next_page(self):
        """Navigate to next page"""
        if self.pdf_document and self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_page_display()
    
    def extract_text(self):
        """Extract text from current PDF page"""
        if not self.pdf_document:
            messagebox.showwarning("No PDF", "Please select a PDF file first")
            return
        
        def extract_text_thread():
            try:
                self.status_var.set("Extracting text...")
                
                # Extract text from current page
                page = self.pdf_document[self.current_page]
                text = page.get_text()
                
                # Store original text
                self.original_text = text
                self.extracted_text = text  # Keep for compatibility
                
                # Update text display
                self.root.after(0, lambda: self.update_text_display(text, "original"))
                self.root.after(0, lambda: self.status_var.set("Text extraction completed"))
                
            except Exception as e:
                error_msg = f"Error extracting text: {str(e)}"
                self.root.after(0, lambda: self.status_var.set(error_msg))
                logger.error(error_msg)
        
        threading.Thread(target=extract_text_thread, daemon=True).start()
    
    def update_text_display(self, text, language="original"):
        """Update the text display widget"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, text)
        
        # Update language indicator
        if language == "original":
            self.text_language_label.config(text="Language: Original (English)")
        elif language == "kannada":
            self.text_language_label.config(text="Language: ಕನ್ನಡ (Kannada)")
        else:
            self.text_language_label.config(text=f"Language: {language}")
        
        self.current_display_language = language
    
    def detect_objects(self):
        """Detect objects in current PDF page"""
        if not self.pdf_document:
            messagebox.showwarning("No PDF", "Please select a PDF file first")
            return
        
        def detect_objects_thread():
            try:
                self.status_var.set("Detecting objects...")
                
                # Convert PDF page to image with multiple fallback methods
                page = self.pdf_document[self.current_page]
                img = None
                
                # Method 1: Try using PyMuPDF's built-in PNG conversion
                try:
                    mat = fitz.Matrix(1.5, 1.5)
                    pix = page.get_pixmap(matrix=mat)
                    logger.info(f"Pixmap info: width={pix.width}, height={pix.height}, n={pix.n}")
                    
                    # Get PNG bytes and load with OpenCV
                    png_data = pix.tobytes("png")
                    img_array = np.frombuffer(png_data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        logger.info(f"Method 1 (PNG): Successfully loaded image with shape {img.shape}")
                    else:
                        raise ValueError("PNG decode failed")
                        
                except Exception as e1:
                    logger.warning(f"Method 1 (PNG) failed: {str(e1)}")
                    
                    # Method 2: Try direct pixel manipulation with proper size handling
                    try:
                        mat = fitz.Matrix(1.0, 1.0)  # Lower resolution to avoid issues
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Get raw bytes
                        img_data = pix.tobytes()
                        data_size = len(img_data)
                        expected_size = pix.width * pix.height * pix.n
                        
                        logger.info(f"Method 2: Data size={data_size}, Expected={expected_size}, Channels={pix.n}")
                        
                        if pix.n == 3:  # RGB
                            # Calculate actual dimensions based on data size
                            pixels_per_channel = data_size // 3
                            actual_height = int(np.sqrt(pixels_per_channel * pix.height / pix.width))
                            actual_width = pixels_per_channel // actual_height
                            
                            if actual_width * actual_height * 3 <= data_size:
                                img_array = np.frombuffer(img_data[:actual_width*actual_height*3], dtype=np.uint8)
                                img = img_array.reshape(actual_height, actual_width, 3)
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                logger.info(f"Method 2: Successfully loaded image with shape {img.shape}")
                            else:
                                raise ValueError("Size calculation failed")
                        else:
                            raise ValueError(f"Unsupported channel count: {pix.n}")
                            
                    except Exception as e2:
                        logger.warning(f"Method 2 (direct) failed: {str(e2)}")
                        
                        # Method 3: Save to temporary file and reload
                        try:
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                                mat = fitz.Matrix(1.0, 1.0)
                                pix = page.get_pixmap(matrix=mat)
                                pix.save(tmp_file.name)
                                
                                # Load with OpenCV
                                img = cv2.imread(tmp_file.name)
                                
                                # Clean up temp file
                                os.unlink(tmp_file.name)
                                
                                if img is not None:
                                    logger.info(f"Method 3 (temp file): Successfully loaded image with shape {img.shape}")
                                else:
                                    raise ValueError("Temp file method failed")
                                    
                        except Exception as e3:
                            logger.error(f"Method 3 (temp file) failed: {str(e3)}")
                            raise ValueError("All image conversion methods failed")
                
                # If we got here, one of the methods worked
                if img is None:
                    raise ValueError("Failed to convert PDF page to image")
                
                # Perform object detection
                detected_objects = self.perform_object_detection(img)
                
                # Store detection results and scale coordinates for display
                self.detected_objects = self.scale_detection_coordinates(detected_objects, img, pix)
                
                # Update objects display
                self.root.after(0, lambda: self.update_objects_display(self.detected_objects))
                self.root.after(0, lambda: self.update_page_display())  # Refresh to show boxes
                self.root.after(0, lambda: self.status_var.set(f"Object detection completed - {len(self.detected_objects)} objects found"))
                
            except Exception as e:
                error_msg = f"Error detecting objects: {str(e)}"
                self.root.after(0, lambda: self.status_var.set(error_msg))
                logger.error(error_msg)
        
        threading.Thread(target=detect_objects_thread, daemon=True).start()
    
    def perform_object_detection(self, image):
        """Perform object detection using computer vision techniques"""
        try:
            # Handle different image formats and fix reshaping issue
            if len(image.shape) == 3:
                height, width, channels = image.shape
            else:
                height, width = image.shape
                channels = 1
            
            logger.info(f"Processing image: {width}x{height}x{channels}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image
            
            detected_objects = []
            
            # Text region detection using MSER (Maximally Stable Extremal Regions)
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            for i, region in enumerate(regions[:20]):  # Limit to top 20 regions
                if len(region) > 10:  # Filter very small regions
                    x, y, w, h = cv2.boundingRect(region)
                    area = w * h
                    
                    if area > 500 and w > 20 and h > 10:  # Filter by size
                        detected_objects.append({
                            'id': i,
                            'class': 'text_region',
                            'confidence': 0.75,
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'area': int(area)
                        })
            
            # Table detection using horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            horizontal_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            vertical_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Add table structures as detected objects
            table_id = len(detected_objects)
            for contour in horizontal_contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_objects.append({
                        'id': table_id,
                        'class': 'table_line',
                        'confidence': 0.80,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': int(area)
                    })
                    table_id += 1
            
            # Image/figure detection using edge detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Larger areas likely to be figures/images
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter based on aspect ratio and size
                    if 0.3 < aspect_ratio < 3.0 and w > 50 and h > 50:
                        detected_objects.append({
                            'id': len(detected_objects),
                            'class': 'figure_or_image',
                            'confidence': 0.70,
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'area': int(area)
                        })
            
            logger.info(f"Detected {len(detected_objects)} objects")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return []
    
    def scale_detection_coordinates(self, detected_objects, detection_img, original_pix):
        """Scale detection coordinates to match display image"""
        try:
            # Get dimensions
            detection_height, detection_width = detection_img.shape[:2]
            original_width = original_pix.width
            original_height = original_pix.height
            
            # Calculate scaling factors
            scale_x = original_width / detection_width
            scale_y = original_height / detection_height
            
            logger.info(f"Scaling detection coordinates: "
                       f"detection({detection_width}x{detection_height}) -> "
                       f"display({original_width}x{original_height})")
            
            # Scale each object's bounding box
            scaled_objects = []
            for obj in detected_objects:
                scaled_obj = obj.copy()
                bbox = obj['bbox']
                
                # Scale coordinates
                scaled_bbox = [
                    int(bbox[0] * scale_x),  # x
                    int(bbox[1] * scale_y),  # y
                    int(bbox[2] * scale_x),  # width
                    int(bbox[3] * scale_y)   # height
                ]
                
                scaled_obj['bbox'] = scaled_bbox
                scaled_objects.append(scaled_obj)
            
            return scaled_objects
            
        except Exception as e:
            logger.error(f"Error scaling coordinates: {str(e)}")
            return detected_objects
        """Update the objects display widget"""
        self.detected_objects = objects
        self.objects_display.delete(1.0, tk.END)
        
        if not objects:
            self.objects_display.insert(tk.END, "No objects detected")
            return
        
        for obj in objects:
            obj_text = (f"Object {obj['id']}: {obj['class']}\n"
                       f"Confidence: {obj['confidence']:.2f}\n"
                       f"Position: ({obj['bbox'][0]}, {obj['bbox'][1]})\n"
                       f"Size: {obj['bbox'][2]}x{obj['bbox'][3]}\n"
                       f"Area: {obj['area']}\n\n")
            self.objects_display.insert(tk.END, obj_text)
    
    def play_tts(self, language):
        """Play text-to-speech in specified language"""
        # Determine which text to use based on current display
        if language == "kn" and self.translated_text and self.current_display_language == "kannada":
            text_to_speak = self.translated_text
        elif language == "en" and self.current_display_language == "original":
            text_to_speak = self.original_text or self.extracted_text
        elif language == "kn" and not self.translated_text:
            messagebox.showwarning("No Kannada Text", "Please translate to Kannada first, or the system will use English pronunciation")
            text_to_speak = self.extracted_text
        else:
            text_to_speak = self.extracted_text
        
        if not text_to_speak.strip():
            messagebox.showwarning("No Text", "Please extract text first")
            return
        
        def tts_thread():
            try:
                method = self.tts_method.get() if hasattr(self, 'tts_method') else "system"
                self.status_var.set(f"Playing audio in {language} using {method} TTS...")
                
                # Clean text for better TTS
                clean_text = self.clean_text_for_tts(text_to_speak)
                
                if language == "kn" and method == "google" and self.gtts_available:
                    # Use Google TTS for Kannada
                    self.play_google_tts(clean_text, language)
                else:
                    # Use system TTS
                    self.play_system_tts(clean_text, language)
                
                self.root.after(0, lambda: self.status_var.set("Audio playback completed"))
                
            except Exception as e:
                error_msg = f"Error playing audio: {str(e)}"
                self.root.after(0, lambda: self.status_var.set(error_msg))
                logger.error(error_msg)
        
        threading.Thread(target=tts_thread, daemon=True).start()
    
    def play_system_tts(self, text, language):
        """Play TTS using system voices"""
        try:
            # Set voice based on language
            if language == "en" and self.english_voice:
                self.tts_engine.setProperty('voice', self.english_voice)
                logger.info(f"Using English voice: {self.english_voice}")
            elif language == "kn":
                if self.kannada_voice and self.kannada_voice != self.english_voice:
                    self.tts_engine.setProperty('voice', self.kannada_voice)
                    logger.info(f"Using Kannada voice: {self.kannada_voice}")
                else:
                    # Fallback: Use English voice but inform user
                    self.tts_engine.setProperty('voice', self.english_voice)
                    logger.warning("No native Kannada voice available, using English voice")
                    self.root.after(0, lambda: messagebox.showinfo(
                        "TTS Method Recommendation", 
                        "System Kannada TTS not available.\n\n"
                        "For better Kannada support:\n"
                        "1. Switch TTS Method to 'google'\n"
                        "2. Or install: pip install gtts pygame\n"
                        "3. Requires internet connection\n\n"
                        "Currently using English voice as fallback."
                    ))
            else:
                # Use default voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Speak the text
            self.tts_playing = True
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.tts_playing = False
            
        except Exception as e:
            logger.error(f"Error in system TTS: {str(e)}")
            raise
    
    def play_google_tts(self, text, language):
        """Play TTS using Google Text-to-Speech"""
        try:
            if not self.gtts_available:
                raise ValueError("Google TTS not available")
            
            # Map language codes
            lang_map = {
                'en': 'en',
                'kn': 'kn'  # Kannada is supported by Google TTS
            }
            
            gtts_lang = lang_map.get(language, 'en')
            logger.info(f"Using Google TTS for language: {gtts_lang}")
            
            # Check internet connection
            if not self.check_internet_connection():
                raise ValueError("Internet connection required for Google TTS")
            
            # Create TTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                temp_audio_path = tmp_file.name
            
            # Play the audio file
            self.tts_playing = True
            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy() and self.tts_playing:
                threading.Event().wait(0.1)
            
            self.tts_playing = False
            
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            logger.info("Google TTS playback completed")
            
        except Exception as e:
            logger.error(f"Error in Google TTS: {str(e)}")
            # Fallback to system TTS
            logger.info("Falling back to system TTS")
            self.play_system_tts(text, language)
    
    def translate_to_kannada(self):
        """Translate extracted text to Kannada"""
        if not self.original_text.strip():
            messagebox.showwarning("No Text", "Please extract text first")
            return
        
        def translation_thread():
            try:
                self.status_var.set("Translating to Kannada...")
                
                if not self.translator:
                    raise ValueError("Translation service not available. Install: pip install googletrans==4.0.0-rc1")
                
                # Check internet connection for Google Translate
                if not self.check_internet_connection():
                    raise ValueError("Internet connection required for translation")
                
                # Clean text for better translation
                clean_text = self.clean_text_for_translation(self.original_text)
                
                # Translate to Kannada
                translated = self.translator.translate(clean_text, src='en', dest='kn')
                kannada_text = translated.text
                
                # Store translated text
                self.translated_text = kannada_text
                
                # Update display
                self.root.after(0, lambda: self.update_text_display(kannada_text, "kannada"))
                self.root.after(0, lambda: self.status_var.set("Translation to Kannada completed"))
                
                logger.info("Translation to Kannada completed successfully")
                
            except Exception as e:
                error_msg = f"Translation error: {str(e)}"
                self.root.after(0, lambda: self.status_var.set(error_msg))
                self.root.after(0, lambda: messagebox.showerror("Translation Error", 
                    f"Failed to translate text:\n{str(e)}\n\n"
                    "Make sure you have:\n"
                    "1. Internet connection\n"
                    "2. Installed: pip install googletrans==4.0.0-rc1"))
                logger.error(error_msg)
        
        threading.Thread(target=translation_thread, daemon=True).start()
    
    def show_original_text(self):
        """Show original extracted text"""
        if self.original_text:
            self.extracted_text = self.original_text
            self.update_text_display(self.original_text, "original")
            self.status_var.set("Showing original text")
        else:
            messagebox.showwarning("No Text", "No original text available")
    
    def show_kannada_text(self):
        """Show Kannada translated text"""
        if self.translated_text:
            self.extracted_text = self.translated_text
            self.update_text_display(self.translated_text, "kannada")
            self.status_var.set("Showing Kannada text")
        else:
            messagebox.showwarning("No Translation", "Please translate to Kannada first")
    
    def clean_text_for_translation(self, text):
        """Clean text for better translation"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove or replace problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Remove page numbers and common PDF artifacts
        import re
        text = re.sub(r'\b\d+\b(?=\s*$)', '', text, flags=re.MULTILINE)  # Remove page numbers at end of lines
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Limit text length for translation API limits
        if len(text) > 4500:  # Google Translate has character limits
            text = text[:4500] + "..."
            logger.warning("Text truncated for translation due to length limits")
        
        return text.strip()
        """Check if internet connection is available"""
        try:
            if not REQUESTS_AVAILABLE:
                return False
            response = requests.get("https://www.google.com", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def stop_tts(self):
        """Stop current TTS playback"""
        try:
            self.tts_playing = False
            
            # Stop system TTS
            if self.tts_engine:
                self.tts_engine.stop()
            
            # Stop Google TTS (pygame)
            if self.gtts_available and pygame.mixer.get_init():
                pygame.mixer.music.stop()
            
            self.status_var.set("Audio playback stopped")
            logger.info("TTS playback stopped")
            
        except Exception as e:
            logger.error(f"Error stopping TTS: {str(e)}")
    
    def clean_text_for_tts(self, text):
        """Clean extracted text for better TTS output"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove or replace problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Limit text length for reasonable audio duration
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text
    
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pdf_document:
                self.pdf_document.close()
            if self.tts_engine:
                self.tts_engine.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Additional utility functions for edge AI processing

class ModelDownloader:
    """Utility class to download and manage AI models"""
    
    @staticmethod
    def download_object_detection_model():
        """Download a lightweight object detection model for edge deployment"""
        # This would download models like YOLOv8n, MobileNet, or other edge-optimized models
        # Models should be converted to ONNX format for optimal performance
        pass
    
    @staticmethod
    def download_text_detection_model():
        """Download OCR/text detection model for better text extraction"""
        # This could include models like PaddleOCR, TrOCR, or other edge-optimized OCR models
        pass

class EdgeOptimizer:
    """Utility class for edge deployment optimizations"""
    
    @staticmethod
    def optimize_onnx_model(model_path, output_path):
        """Optimize ONNX model for edge deployment"""
        try:
            import onnxoptimizer
            
            # Load model
            model = onnx.load(model_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model, [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_monotone_argmax',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_log_softmax',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ])
            
            # Save optimized model
            onnx.save(optimized_model, output_path)
            logger.info(f"Model optimized and saved to {output_path}")
            
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")

# Add missing import
import io

if __name__ == "__main__":
    # Create and run the application
    app = EdgePDFProcessor()
    app.run()
