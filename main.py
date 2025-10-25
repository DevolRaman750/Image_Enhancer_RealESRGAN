import warnings
import os
from glob import glob

import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

warnings.filterwarnings("ignore")

# Paths
model_path = 'RealESRGAN_x4plus.pth'
input_folder = 'input_images'
output_folder = 'output_images'

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load model
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)

# Initialize upsampler (CPU mode here; set half=True if you later use CUDA GPU)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    half=False
)

# Get all image files (jpg, png, jpeg)
image_files = glob(os.path.join(input_folder, '*.*'))
valid_exts = ('.jpg', '.jpeg', '.png')

for img_path in image_files:
    if img_path.lower().endswith(valid_exts):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)

            # Enhance
            output, _ = upsampler.enhance(img, outscale=4)

            # Save output
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_folder, filename)
            Image.fromarray(output).save(save_path)

            print(f"✅ Processed: {filename} -> {save_path}")

        except Exception as e:
            print(f"❌ Failed on {img_path}: {e}")

print("\n🎉 All images processed. Check 'output_images' folder.")
