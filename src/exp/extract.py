import pymupdf
from constants import DATA_DIR
import json
from collections import defaultdict

IMAGE_OUTPUT_DIR = DATA_DIR / "extracted_images"
PDF_OUTPUT_DIR = DATA_DIR / "extracted_pdf"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Helper function to extract images and save
def extract_images_from_page(doc, page_no):
    images = []
    for img_index, img in enumerate(doc.get_page_images(page_no, full=True)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_filename = f"page_{page_no + 1}_img_{img_index + 1}.{image_ext}"
        image_filepath = IMAGE_OUTPUT_DIR / image_filename
        with open(image_filepath, "wb") as img_file:
            img_file.write(image_bytes)
        images.append({
            "xref": xref,
            "image_file": str(image_filepath),
            "width": base_image["width"],
            "height": base_image["height"]
        })
    return images


def extract_pdf(filename: str):
    # Load Document
    doc = pymupdf.open(DATA_DIR / "pdf" / filename)

    toc = doc.get_toc()
    metadata = doc.metadata

    toc_dict = defaultdict(list)
    for entry in toc:
        depth, title, page_no = entry
        toc_dict[page_no - 1].append({'depth': depth, 'title': title})


    # Build structured representation
    document_structure = {
        "metadata": metadata,
        "toc": toc,
        "pages": []
    }


    # Scan all pages and extract content
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)
        images = extract_images_from_page(doc, pno)

        content_blocks = []
        for block in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            content_blocks.append({
                "type": "text",
                "bbox": [x0, y0, x1, y1],
                "text": text.strip()
            })

        for img in images:
            content_blocks.append({
                "type": "image",
                "image_file": img["image_file"],
                "size": [img["width"], img["height"]]
            })

        page_data = {
            "page_number": pno + 1,
            "toc_titles": toc_dict.get(pno, []),
            "content": content_blocks
        }

        document_structure["pages"].append(page_data)


    # Optionally detect and flag "References" section
    for page_data in document_structure["pages"]:
        for block in page_data["content"]:
            if block["type"] == "text" and "references" in block["text"].lower():
                page_data["is_reference_section"] = True
                break


    with open(PDF_OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(document_structure, f, indent=2, ensure_ascii=False)

    return document_structure