import pymupdf
from pathlib import Path

FILEPATH = Path().cwd().joinpath("docs", "master-degree.pdf")
FILEPATH = Path().cwd().joinpath("docs", "LDAC2024_CFP.pdf")
FILEPATH = Path().cwd().joinpath("docs", "thesis.pdf")


def main():

    # doc = pymupdf.open(FILEPATH)
    # out = open("output/master-degree.txt", "wb")
    # for page in doc:  # iterate the document pages
    #     text = page.get_text().encode("utf8")
    #     out.write(text)  # write text of page
    #     out.write(bytes((12,)))  # write page delimiter (form feed 0x0C)
    # out.close()

    # doc = pymupdf.open(FILEPATH)  # open a document
    #
    # for page_index in range(len(doc)):  # iterate over pdf pages
    #     page = doc[page_index]  # get the page
    #     image_list = page.get_images()
    #
    #     # print the number of images found on the page
    #     if image_list:
    #         print(f"Found {len(image_list)} images on page {page_index}")
    #     else:
    #         print("No images found on page", page_index)
    #
    #     for image_index, img in enumerate(image_list,
    #                                       start=1):  # enumerate the image list
    #         xref = img[0]  # get the XREF of the image
    #         pix = pymupdf.Pixmap(doc, xref)  # create a Pixmap
    #
    #         if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
    #             pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    #
    #         pix.save("output/page_%s-image_%s.png" % (page_index,
    #                                            image_index))  # save the image as png
    #         pix = None

    doc = pymupdf.open(FILEPATH)
    toc = doc.get_toc()


    print('_')


if __name__ == "__main__":
    main()