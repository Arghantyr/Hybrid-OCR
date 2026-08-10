# 1. module imports
from abc import ABC, abstractmethod
from typing import Dict, List, Type
from PIL import Image

# 2. constant definitions
DEFAULT_DEVICE = "cuda"
DEFAULT_PROMPT = "Transcribe the text in this document line by line exactly."

# 3. class definitions
class BaseOCRModel(ABC):
    def __init__(self, model_id: str, device: str = DEFAULT_DEVICE):
        self.model_id = model_id
        self.device = device

    @abstractmethod
    def transcribe_images(self, images: List[Image.Image]) -> str:
        pass

class DummyOCRModel(BaseOCRModel):
    def __init__(self, model_id: str = "dummy-ocr-v1", device: str = DEFAULT_DEVICE):
        super().__init__(model_id, device)

    def transcribe_images(self, images: List[Image.Image]) -> str:
        return "1. i-na a-lim(KI) KÀ-ŠU-UM\n2. DUMU ša-lim-a-ḫum\n3. i-na É a-ba-am\n4. IŠ-TÚ KÙ.BABBAR 10 MA.NA\n5. KÀ-RU-UM kà-ni-iš"

class TrOCRModel(BaseOCRModel):
    def __init__(self, model_id: str = "microsoft/trocr-base-printed", device: str = DEFAULT_DEVICE):
        super().__init__(model_id, device)
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.processor = TrOCRProcessor.from_pretrained(self.model_id, use_fast=False)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            if self.device != "cpu":
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    pass

    def transcribe_images(self, images: List[Image.Image]) -> str:
        self._load_model()
        results = []
        for img in images:
            try:
                pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
                if self.device != "cpu":
                    try:
                        pixel_values = pixel_values.to(self.device)
                    except Exception:
                        pass
                generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results.append(text)
            except Exception as e:
                results.append(f"[Error running TrOCR: {e}]")
        return "\n".join(results)

class GLMOCRModel(BaseOCRModel):
    def __init__(self, model_id: str = "zai-org/GLM-OCR", device: str = DEFAULT_DEVICE):
        super().__init__(model_id, device)
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            from transformers import AutoProcessor, AutoModelForCausalLM
            try:
                from transformers import GlmOcrForConditionalGeneration as ModelClass
            except ImportError:
                ModelClass = AutoModelForCausalLM

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = ModelClass.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
            )

    def transcribe_images(self, images: List[Image.Image]) -> str:
        self._load_model()
        results = []
        for img in images:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Text Recognition:"},
                        ],
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt")
                if self.device != "cpu":
                    try:
                        inputs = inputs.to(self.device)
                    except Exception:
                        pass
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                results.append(output_text.strip())
            except Exception as e:
                results.append(f"[Error running zai-org/GLM-OCR: {e}]")
        return "\n".join(results)

class Qwen2VLOCRModel(BaseOCRModel):
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = DEFAULT_DEVICE):
        super().__init__(model_id, device)
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto"
            )

    def transcribe_images(self, images: List[Image.Image]) -> str:
        self._load_model()
        results = []
        for img in images:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": DEFAULT_PROMPT},
                        ],
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt")
                if self.device != "cpu":
                    try:
                        inputs = inputs.to(self.device)
                    except Exception:
                        pass
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                results.append(output_text.strip())
            except Exception as e:
                results.append(f"[Error running Qwen2-VL: {e}]")
        return "\n".join(results)

class GOTOCRModel(BaseOCRModel):
    def __init__(self, model_id: str = "stepfun-ai/GOT-OCR2_0", device: str = DEFAULT_DEVICE):
        super().__init__(model_id, device)
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            from transformers import AutoTokenizer, AutoModel
            try:
                from transformers import GotOcr2ForConditionalGeneration as ModelClass
            except ImportError:
                ModelClass = AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = ModelClass.from_pretrained(
                self.model_id, trust_remote_code=True, device_map="auto"
            )

    def transcribe_images(self, images: List[Image.Image]) -> str:
        self._load_model()
        results = []
        for img in images:
            try:
                res = self.model.chat(self.tokenizer, img, ocr_type="format")
                results.append(str(res).strip())
            except Exception as e:
                results.append(f"[Error running GOT-OCR: {e}]")
        return "\n".join(results)

class DonutOCRModel(BaseOCRModel):
    def __init__(self, model_id: str = "naver-clova-ix/donut-base", device: str = DEFAULT_DEVICE):
        super().__init__(model_id, device)
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            self.processor = DonutProcessor.from_pretrained(self.model_id)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            if self.device != "cpu":
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    pass

    def transcribe_images(self, images: List[Image.Image]) -> str:
        self._load_model()
        results = []
        for img in images:
            try:
                pixel_values = self.processor(img, return_tensors="pt").pixel_values
                if self.device != "cpu":
                    try:
                        pixel_values = pixel_values.to(self.device)
                    except Exception:
                        pass
                decoder_input_ids = self.processor.tokenizer(
                    "<s_docvqa><s_question>Transcribe text line by line</s_question><s_answer>",
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids
                if self.device != "cpu":
                    try:
                        decoder_input_ids = decoder_input_ids.to(self.device)
                    except Exception:
                        pass
                outputs = self.model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
                seq = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                results.append(seq.strip())
            except Exception as e:
                results.append(f"[Error running Donut: {e}]")
        return "\n".join(results)

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Type[BaseOCRModel]] = {
            "dummy": DummyOCRModel,
            "trocr": TrOCRModel,
            "glm-ocr": GLMOCRModel,
            "qwen2-vl": Qwen2VLOCRModel,
            "got-ocr": GOTOCRModel,
            "donut": DonutOCRModel,
        }

    def register(self, name: str, model_cls: Type[BaseOCRModel]):
        self._models[name.lower()] = model_cls

    def get_model(self, name: str, model_id: str = None, device: str = DEFAULT_DEVICE) -> BaseOCRModel:
        cls = self._models.get(name.lower())
        if not cls:
            raise ValueError(f"Unknown model name: '{name}'. Available: {list(self._models.keys())}")
        return cls(model_id=model_id, device=device) if model_id else cls(device=device)

# 4. function definitions
def get_available_models() -> List[str]:
    return list(global_model_registry._models.keys())

# 5. class instantiation
global_model_registry = ModelRegistry()
