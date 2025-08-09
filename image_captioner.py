import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time

class ImageCaptioner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("JoyCaption Image Captioner")
        self.root.geometry("800x600")
        
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        self.selected_folder = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.caption_style = tk.StringVar(value="descriptive")
        self.overwrite_existing = tk.BooleanVar(value=False)
        
        self.model_path = "llama-joycaption-beta-one-hf-llava"
        
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="JoyCaption Image Captioner", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Folder selection
        ttk.Label(main_frame, text="Select Image Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.selected_folder, width=50).grid(row=1, column=1, padx=(10, 5), pady=5, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.select_folder).grid(row=1, column=2, pady=5)
        
        # Caption style selection
        ttk.Label(main_frame, text="Caption Style:").grid(row=2, column=0, sticky=tk.W, pady=5)
        style_frame = ttk.Frame(main_frame)
        style_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        styles = [
            ("descriptive", "Descriptive (Detailed descriptions)"),
            ("straightforward", "Straightforward (Concise, objective)"),
            ("training", "Training Style (For LoRA training)")
        ]
        
        for i, (value, text) in enumerate(styles):
            ttk.Radiobutton(style_frame, text=text, variable=self.caption_style, value=value).grid(row=0, column=i, sticky=tk.W, padx=(0, 20))
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(0, weight=1)
        
        ttk.Checkbutton(options_frame, text="Overwrite existing caption files", 
                       variable=self.overwrite_existing).grid(row=0, column=0, sticky=tk.W)
        
        # Model status
        model_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        model_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.model_status_label = ttk.Label(model_frame, text="Not loaded", foreground="red")
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model_async).grid(row=0, column=2, padx=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Captioning", command=self.start_captioning, state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_captioning, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=1, column=0, sticky=tk.W)
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.stop_processing = False
        
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing images")
        if folder:
            self.selected_folder.set(folder)
            self.log_message(f"Selected folder: {folder}")
            
    def load_model_async(self):
        threading.Thread(target=self.load_model, daemon=True).start()
        
    def load_model(self):
        try:
            self.model_status_label.config(text="Loading...", foreground="orange")
            self.log_message("Loading JoyCaption model...")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            self.model.eval()
            
            self.model_loaded = True
            self.model_status_label.config(text="Loaded", foreground="green")
            self.start_button.config(state="normal")
            self.log_message("Model loaded successfully!")
            
        except Exception as e:
            self.model_status_label.config(text="Error", foreground="red")
            self.log_message(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def get_caption_prompt(self, style):
        if style == "descriptive":
            return "Write a long detailed description for this image."
        elif style == "straightforward":
            return ("Write a straightforward caption for this image. Begin with the main subject and medium. "
                   "Mention pivotal elements—people, objects, scenery—using confident, definite language. "
                   "Focus on concrete details like color, shape, texture, and spatial relationships. "
                   "Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. "
                   "Note any watermarks, signatures, or compression artifacts. Never mention what's absent, "
                   "resolution, or unobservable details. Vary your sentence structure and keep the description concise, "
                   "without starting with 'This image is…' or similar phrasing.")
        elif style == "training":
            return ("Write a short, simple caption describing this image. Use only factual, concrete details. "
                   "Avoid atmospheric words, mood descriptions, or ambiguous language. Keep it brief and direct. "
                   "Focus on what is clearly visible: subjects, objects, actions, basic colors, and clear visual elements. "
                   "Do not use phrases like 'This image shows' or 'The photo depicts'. "
                   "Avoid words that could have multiple meanings or interpretations.")
        else:
            return "Write a long detailed description for this image."
            
    def caption_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            
            prompt = self.get_caption_prompt(self.caption_style.get())
            

            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user", 
                    "content": prompt,
                },
            ]
            
            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    suppress_tokens=None,
                    use_cache=True,
                    temperature=0.6,
                    top_k=None,
                    top_p=0.9,
                )[0]
                
                generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
                
                caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                caption = caption.strip()
                
            return caption
            
        except Exception as e:
            self.log_message(f"Error captioning {image_path}: {str(e)}")
            return None
            
    def start_captioning(self):
        if not self.selected_folder.get():
            messagebox.showerror("Error", "Please select a folder containing images")
            return
            
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load the model first")
            return
            
        self.stop_processing = False
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self.process_images, daemon=True).start()
        
    def stop_captioning(self):
        self.stop_processing = True
        self.log_message("Stopping captioning process")
        
    def process_images(self):
        try:
            folder_path = Path(self.selected_folder.get())
            
            image_files = []
            for ext in self.supported_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
                
            if not image_files:
                self.log_message("No image files found in the selected folder")
                return
                
            self.log_message(f"Found {len(image_files)} image files")
            total_files = len(image_files)
            processed = 0
            
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            for i, image_file in enumerate(image_files):
                if self.stop_processing:
                    break
                    
                caption_file = image_file.with_suffix('.txt')
                if caption_file.exists() and not self.overwrite_existing.get():
                    self.log_message(f"Skipping {image_file.name} caption already exists")
                    processed += 1
                    continue
                    
                self.status_var.set(f"Processing {image_file.name}")
                self.log_message(f"Captioning {image_file.name}")
                
                caption = self.caption_image(image_file)
                
                if caption:
                    try:
                        with open(caption_file, 'w', encoding='utf-8') as f:
                            f.write(caption)
                        self.log_message(f"Saved caption for {image_file.name}")
                    except Exception as e:
                        self.log_message(f"Error saving caption for {image_file.name}: {str(e)}")
                
                processed += 1
                progress = (processed / total_files) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
                
            if self.stop_processing:
                self.status_var.set("Captioning stopped")
                self.log_message("Captioning process stopped by user")
            else:
                self.status_var.set("Captioning completed")
                self.log_message(f"Captioning completed! Processed {processed} images")
                
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Error during processing: {str(e)}")
            
        finally:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
    def run(self):
        self.log_message("JoyCaption Image Captioner started")
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageCaptioner()
    app.run()
