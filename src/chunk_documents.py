def chunk_text(text,chunk_size=200,overlap=50):
    words=text.split()
    chunks=[]

    start=0
    while start<len(words):
        end=start+chunk_size
        chunk_words=words[start:end]
        chunk=" ".join(chunk_words)
        chunks.append(chunk)
        start+=chunk_size-overlap
    
    return chunks


def chunk_documents(documents, chunk_size=200,overlap=50 ):
    all_chunks=[]

    for doc in documents:
        text_chunks=chunk_text(doc["text"],chunk_size,overlap)
        for i,chunk in enumerate(text_chunks):
            all_chunks.append({
                "text":chunk,
                "source":doc["source"],
                "page":doc["page"],
                "chunk_id":f"{doc['source']}_p{doc['page']}_c{i}"
            })

    return all_chunks






if __name__=="__main__":
    from load_documents import load_pdfs

    docs=load_pdfs("../data")
    chunks=chunk_documents(docs)

    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    print(len(chunks))