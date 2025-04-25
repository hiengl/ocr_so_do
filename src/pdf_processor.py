import fitz  # pymupdf
import os


def pdf_to_images(pdf_path, output_dir, dpi=300):
    """Convert PDF to images."""
    pdf_document = fitz.open(pdf_path)
    image_paths = []

    os.makedirs(output_dir, exist_ok=True)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(output_path)
        image_paths.append(output_path)

    pdf_document.close()
    return image_paths