import os
from pypdf import PdfReader

def load_pdfs(folder_path):
    all_documents=[]

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath=os.path.join(folder_path,filename)
            reader=PdfReader(filepath)

            for page_num,page in enumerate(reader.pages):
                text=page.extract_text()
                if text.strip():
                    all_documents.append({
                        "text":text,
                        "source":filename,
                        "page":page_num+1
                    })
    return all_documents


if __name__=="__main__":
    docs=load_pdfs("../data")
    print(f"Loaded {len(docs)} pages total")
    print(docs[2])