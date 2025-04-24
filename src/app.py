import streamlit as st
from pdf_processor import pdf_to_images
from ocr_processor import OCRProcessor
from text_formatter import TextFormatter
from utils import load_config, save_json
import os
from pathlib import Path
import json

# Inject custom CSS for independent scrolling
st.markdown(
    """
    <style>
    .scrollable-column {
        max-height: 600px;  /* Adjust height as needed */
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;  /* Optional: for visual separation */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def display_field(field, value, key_prefix: str, level: int = 0):
    """Recursively display and edit a field in the JSON structure."""
    indent = "  " * level
    field_key = f"{key_prefix}_{field['key']}"

    if field["type"] == "string" or field["type"] == "string_or_null":
        # Display string or nullable string field
        st.text_input(f"{indent}{field['name']}", value=value or "", key=field_key)
        return {field["key"]: st.session_state[field_key]}

    elif field["type"] == "object" or field["type"] == "object_or_null":
        # Display object or nullable object
        if value is None and field["type"] == "object_or_null":
            if st.checkbox(f"{indent}{field['name']} (Include)", key=f"{field_key}_toggle"):
                value = {subfield["key"]: "" for subfield in field.get("subfields", [])}
            else:
                return {field["key"]: None}
        st.write(f"{indent}{field['name']}:")
        obj_result = {}
        for subfield in field.get("subfields", []):
            sub_value = value.get(subfield["key"]) if value else ""
            obj_result.update(display_field(subfield, sub_value, f"{field_key}", level + 1))
        return {field["key"]: obj_result}

    elif field["type"] == "array":
        # Display array field
        st.write(f"{indent}{field['name']}:")
        array_result = []
        num_items = len(value) if value else 1
        num_items = st.number_input(
            f"{indent}Số lượng {field['name']}", min_value=0, value=num_items, key=f"{field_key}_count"
        )
        for i in range(num_items):
            st.write(f"{indent}  Item {i + 1}:")
            item_result = {}
            item_value = value[i] if i < len(value) else {subfield["key"]: "" for subfield in field["subfields"]}
            for subfield in field["subfields"]:
                sub_value = item_value.get(subfield["key"]) if item_value else ""
                item_result.update(
                    display_field(subfield, sub_value, f"{field_key}_{i}", level + 2)
                )
            array_result.append(item_result)
        return {field["key"]: array_result}

    return {field["key"]: value}


def main():
    st.title("Bóc tách thông tin Sổ đỏ")

    # Load config
    config = load_config("config/config.yaml")

    # File uploader
    uploaded_file = st.file_uploader("Tải lên file PDF", type="pdf")

    if uploaded_file:
        # Save uploaded file
        pdf_path = os.path.join("data/input", uploaded_file.name)
        os.makedirs("data/input", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process PDF
        with st.spinner("Đang xử lý..."):
            # Step 1: Convert PDF to images
            image_paths = pdf_to_images(pdf_path, config["temp_dir"])

            # Step 2:OCR processing
            ocr_processor = OCRProcessor(config)
            ocr_results = ocr_processor.process_images(image_paths)

            # Step 3: Format text
            formatter = TextFormatter(config)
            formatted_results = formatter.format_text(ocr_results)

        # Display results in two-column layout with independent scrolling
        st.subheader("Kết quả bóc tách")
        edited_results = []

        # Create two columns
        col1, col2 = st.columns([6, 4])

        # Left column: Images in a scrollable container
        with col1:
            with st.container():
                st.markdown('<div class="scrollable-column">', unsafe_allow_html=True)
                for page_num, image_path in enumerate(image_paths, 1):
                    st.write(f"Trang {page_num}")
                    st.image(image_path, caption=f"Trang {page_num}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Right column: Fields in a scrollable container
        with col2:
            with st.container():
                st.markdown('<div class="scrollable-column">', unsafe_allow_html=True)
                edited_result = {}
                for field in config["fields"]:
                    value = formatted_results.get(field["key"])
                    edited_result.update(display_field(field, value, f"page_{page_num}", level=0))

                st.markdown('</div>', unsafe_allow_html=True)

        # Save button
        if st.button("Lưu kết quả"):
            output_path = os.path.join("data/output", f"result_{uploaded_file.name}.json")
            save_json(edited_result, output_path)
            st.success(f"Đã lưu kết quả vào {output_path}")


if __name__ == "__main__":
    main()