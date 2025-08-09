# JoyCaption Batch Image Captioner

<p align="center">
  <img src="https://i.ibb.co/nNdt9YY2/image-2025-08-08-230042321.png" width="400" />
</p>

Created a batch image processing tool for captioning images since JoyCaption's demo only captions one image at a time.

## Installation

1. First clone Llama JoyCaption Beta One

```bash
git clone https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava
```

2. Create venv and activate

```bash
python -m venv venv or py -m venv venv

venv\Scripts\activate
```

3. Install requirements

**For GPU:**

```bash
pip install -r requirements-cuda.txt
```

**For CPU:**

```bash
pip install -r requirements.txt
```

4. Run GUI

```bash
python image_captioner.py or py image_captioner.py
```

5. CLI Alternative

```bash
python batch_caption.py <image_folder> [options]
```

**Examples:**

```bash
python batch_caption.py "C:\MyImages"

# Overwrite existing captions
python batch_caption.py "C:\MyImages" --overwrite

# Use descriptive style
python batch_caption.py "C:\MyImages" --style descriptive

# Custom model path
python batch_caption.py "C:\MyImages" --model "path/to/custom/model"

**Options:**

- `--style` : Caption style (`training`, `descriptive`, `straightforward`) - Default: `training`
- `--overwrite` : Overwrite existing caption files
- `--model` : Custom model path - Default: `llama-joycaption-beta-one-hf-llava`
```

**Caption Styles:**

- `training` (default): Short, factual descriptions perfect for LoRA training
- `descriptive`: Long, detailed descriptions
- `straightforward`: Concise, objective captions

Creates `.txt` files alongside each image.
