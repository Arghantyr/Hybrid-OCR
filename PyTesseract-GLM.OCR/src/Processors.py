class PDFDocument:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    def get_page_as_image(self, page_number, dpi=300):
        """Extracts a page from a PDF as a PIL Image."""
        try:
            doc = pymupdf.open(self.pdf_path)
            # PyMuPDF is 0-indexed
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"PyMuPDF failed. {e}")

class ImageProcessor:
    def __init__(self):
        pass
    def preprocess_image(self, image):
        if image is None:
            print("Error: Could not load image.")
            return None

        img = np.array(image)
        # Handle Grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Prepare inverted binary image for ink detection (Text=255, Bg=0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bin_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_img, w_img = bin_inv.shape

        # --- Preprocessing for Tesseract (Adaptive Threshold) ---
        # Use img_gray as input (must be single channel uint8)
        img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        return img

class TextFieldsprocessor:
    def __init__(self):
        self.fields={}
    def get_fields(self, image, psm=11):
        # 2. Tesseract Layout Analysis
        print("Running Tesseract layout analysis...")
        custom_config = r'' + f'--psm {psm}'
        try:
            return pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
        except Exception as e:
            print(f"Tesseract Error: {e}")

    def group_fields(self):
        # 3. Group Text into Fields
        # Group by (Block, Para, Line) to respect column splits
        lines_map = {}
        n_boxes = len(d['level'])

        for i in range(n_boxes):
            if d['level'][i] != 5: # Word level
                continue
            text = d['text'][i].strip()
            if not text:
                continue

            key = (d['block_num'][i], d['par_num'][i], d['line_num'][i])

            if key not in lines_map:
                lines_map[key] = []

            lines_map[key].append({
                'left': d['left'][i],
                'top': d['top'][i],
                'width': d['width'][i],
                'height': d['height'][i]
            })

        raw_fields = []
        for key, boxes in lines_map.items():
            x1 = min(b['left'] for b in boxes)
            y1 = min(b['top'] for b in boxes)
            x2 = max(b['left'] + b['width'] for b in boxes)
            y2 = max(b['top'] + b['height'] for b in boxes)
            raw_fields.append((x1, y1, x2 - x1, y2 - y1))
        return raw_fields
