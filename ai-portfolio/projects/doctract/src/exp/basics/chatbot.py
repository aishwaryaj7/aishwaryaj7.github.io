import pymupdf
from pathlib import Path
import textwrap
from openai import OpenAI

API_KEY = ""

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    row_count = 0
    header = ""

    for page in doc:
        tables = page.find_tables()
        for table in tables:
            if page.number == 0 and table.header.external:
                header = (
                        ";".join(
                            [
                                name if name is not None else ""
                                for name in table.header.names
                            ]
                        )
                        + "\n"
                )
                text += header
                row_count += 1

            for row in table.extract():
                row_text = (
                        ";".join(
                            [cell if cell is not None else "" for cell in row]) + "\n"
                )
                if row_text != header:
                    text += row_text
                    row_count += 1
    doc.close()
    print(f"Loaded {row_count} table rows from file '{doc.name}'.\n")
    return text


def generate_response_with_chatgpt(client, prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()


FILEPATH = Path().cwd().joinpath("docs", "national-capitals.pdf")

if __name__ == "__main__":
    text = extract_text_from_pdf(FILEPATH)
    client = OpenAI(api_key=API_KEY)

    print("Ready - ask questions or exit with q/Q:")
    while True:
        user_query = input("==> ")

        if user_query.lower().strip() == 'q':
            break
        prompt = text + '\n\n' + user_query
        response = generate_response_with_chatgpt(client, prompt)

        print("Response: \n")

        for line in textwrap.wrap(response, width=70):
            print(line)

        print('-' * 10)

    print('_')