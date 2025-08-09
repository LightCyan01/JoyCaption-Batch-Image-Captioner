import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(model_path):
    print("Loading JoyCaption model")
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully!")
    return processor, model

def get_caption_prompt(style="descriptive"):
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

def caption_image(processor, model, image_path, style="descriptive"):
    try:
        image = Image.open(image_path).convert('RGB')
        
        prompt = get_caption_prompt(style)
        
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
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        with torch.no_grad():
            generate_ids = model.generate(
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
            
            caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            caption = caption.strip()
            
        return caption
        
    except Exception as e:
        print(f"Error captioning {image_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Caption images using JoyCaption")
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--model", default="llama-joycaption-beta-one-hf-llava", 
                       help="Path to JoyCaption model")
    parser.add_argument("--style", choices=["descriptive", "straightforward", "training"], 
                       default="training", help="Caption style")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing caption files")
    
    args = parser.parse_args()
    
    processor, model = load_model(args.model)
    
    folder_path = Path(args.folder)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    image_files = []
    for ext in supported_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("No image files found in the folder")
        return
    
    print(f"Found {len(image_files)} image files")
    
    processed = 0
    for i, image_file in enumerate(image_files):
        caption_file = image_file.with_suffix('.txt')
        
        if caption_file.exists() and not args.overwrite:
            print(f"Skipping {image_file.name} caption already exists")
            continue
        
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        caption = caption_image(processor, model, image_file, args.style)
        
        if caption:
            try:
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)
                print(f"Saved caption for {image_file.name}")
                processed += 1
            except Exception as e:
                print(f"Error saving caption for {image_file.name}: {str(e)}")
    
    print(f"Processing completed! Captioned {processed} images")

if __name__ == "__main__":
    main()
