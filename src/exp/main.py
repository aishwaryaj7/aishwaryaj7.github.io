from src.test.extract import extract_pdf

document_as_json = extract_pdf(filename="paper1.pdf")



print('_')

'''
Document.search_page_for() search for a string on a page

page_text_list = doc.get_page_text(pno=3)
page_image_list = doc.extract_image(xref=73)
page_image_list = doc.get_page_images(pno=3, full=True)
page_pixmap = doc.get_page_pixmap(pno=3)
toc = doc.get_toc()
has_annot = doc.has_annots()
has_link = doc.has_links()

page = doc.load_page(3)
no_chapter = doc.chapter_count
no_chapter = doc.is_form_pdf
no_chapter = doc.is_reflowable
meta = doc.metadata

for page in doc:
    print(page)

'''