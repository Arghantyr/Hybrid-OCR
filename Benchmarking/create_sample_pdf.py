# 1. module imports
import os
from PIL import Image, ImageDraw, ImageFont

# 2. constant definitions
OUTPUT_PDF = "oinput/sample1.pdf"
IMAGE_SIZE = (800, 1000)

# 3. class definitions
# PDF Generator helper

# 4. function definitions
def create_dummy_pdf(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.new("RGB", IMAGE_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "1. i-na a-lim(KI) KA-SU-UM",
        "2. DUMU sa-lim-a-hum",
        "3. i-na E a-ba-am",
        "4. IS-TU KU.BABBAR 10 MA.NA",
        "5. KA-RU-UM ka-ni-is"
    ]
    y = 50
    for line in lines:
        draw.text((50, y), line, fill=(0, 0, 0))
        y += 40
    img.save(output_path, "PDF", resolution=100.0)

# 5. class instantiation
if __name__ == "__main__":
    create_dummy_pdf(OUTPUT_PDF)
    print(f"Created sample PDF at: {OUTPUT_PDF}")
