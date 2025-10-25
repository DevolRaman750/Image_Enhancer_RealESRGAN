# Real-ESRGAN Image Upscaler

This project uses the **Real-ESRGAN model** to upscale and enhance images.  
It automatically processes all images inside the `input_images/` folder and saves the enhanced versions in the `output_images/` folder.

---

## 📂 Project Structure
project/
│── input_images/ # Place your input images here
│── output_images/ # Enhanced images will be saved here
│── RealESRGAN_x4plus.pth # Pre-trained model (download required)
│── main.py # Main Python script
│── requirements.txt # Dependencies
│── README.md # Instructions



---

## Create and activate a virtual environment
  python -m venv venv
  pip install -r requirements.txt

  Download link for RealESRGAN_x4plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.1