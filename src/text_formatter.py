from openai import OpenAI
from typing import List, Dict, Tuple
import json
import os
from .utils import parse_json_from_code_block

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


class TextFormatter:
    def __init__(self, config):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.fields = config["fields"]
        self.schema = self._build_schema()

    def _build_schema(self) -> dict:
        """Build the JSON schema from the field definitions."""

        def build_field(field):
            if field["type"] == "string":
                return {"type": "string"}
            elif field["type"] == "string_or_null":
                return {"type": ["string", "null"]}
            elif field["type"] == "array":
                sub_schema = {"type": "object", "properties": {}, "required": []}
                for subfield in field.get("subfields", []):
                    sub_schema["properties"][subfield["key"]] = build_field(subfield)
                    sub_schema["required"].append(subfield["key"])
                return {"type": "array", "items": sub_schema}
            elif field["type"] == "object":
                object_schema = {"type": "object", "properties": {}, "required": []}
                for subfield in field.get("subfields", []):
                    object_schema["properties"][subfield["key"]] = build_field(subfield)
                    object_schema["required"].append(subfield["key"])
                return object_schema
            elif field["type"] == "object_or_null":
                return {"type": ["object", "null"], "properties": {}, "required": []}
            return {}

        schema = {"type": "object", "properties": {}, "required": []}
        for field in self.fields:
            schema["properties"][field["key"]] = build_field(field)
            schema["required"].append(field["key"])
        return schema

    def _generate_field_descriptions(self, fields, indent_level=0) -> str:
        """Generate detailed descriptions of the fields for the prompt."""
        descriptions = []
        indent = "  " * indent_level
        for field in fields:
            descriptions.append(f"{indent}- Tên: {field['name']} (key: {field['key']}, type: {field['type']})")
            descriptions.append(f"{indent}  Mô tả: {field.get('description', 'Không có mô tả')}")
            if "subfields" in field:
                descriptions.append(f"{indent}  Trường con:")
                descriptions.append(self._generate_field_descriptions(field["subfields"], indent_level + 2))
        return "\n".join(descriptions)

    def format_text(self, ocr_results: List[List[Tuple[str, Tuple[int, int, int, int]]]]) -> List[Dict]:
        """Format OCR results using the OpenAI API."""
        formatted_results = []

        field_descriptions = self._generate_field_descriptions(self.fields)

        for idx, page_result in enumerate(ocr_results, 1):
            # Combine text from the page
            page_text = "\n".join(page_result)
            formatted_results.append(f'--------PAGE {idx}--------\n{page_text}')
        text = "\n\n".join(formatted_results)

        prompt = f"""
            Dựa trên văn bản sau đây được trích xuất từ giấy chứng nhận quyền sử dụng đất của Việt Nam, hãy định dạng nó theo khung dữ liệu JSON được chỉ định.

            **Văn bản:** {text}

            **Khung dữ liệu:** {json.dumps(self.schema, ensure_ascii=False)}

            **Mô tả chi tiết các trường:**
            {field_descriptions}

            Trả về kết quả dưới dạng một đối tượng JSON duy nhất khớp với khung dữ liệu. Đảm bảo tất cả các trường bắt buộc được điền, và các trường không bắt buộc (string_or_null, object_or_null) được đặt thành null nếu không tìm thấy thông tin liên quan. Sử dụng mô tả chi tiết của các trường để xác định thông tin chính xác từ văn bản.
            """
        response = self.client.chat.completions.create(
        model = "gpt-4o",
        messages = [{"role": "user", "content": prompt}],
        temperature=0.3
        )

        return parse_json_from_code_block(response.choices[0].message.content)