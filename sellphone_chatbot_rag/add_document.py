from rag.core import RAG

rag = RAG(
    llm=None
)
# # Thêm dữ liệu
# rag.add_document(
#     doc_id="1",
#     text="Laptop Dell XPS 13, chip Intel Core i7, RAM 16GB, SSD 512GB",
#     metadata={"title": "Laptop Dell XPS 13", "current_price": "30 triệu", "product_promotion": "Giảm 5%"}
# )

# rag.add_document(
#     doc_id="2",
#     text="iPhone 15 Pro Max, màn hình OLED, camera 48MP, pin 5000mAh",
#     metadata={"title": "iPhone 15 Pro Max", "current_price": "35 triệu", "product_promotion": "Tặng sạc nhanh"}
# )
# Lấy danh sách tất cả các documents trong collection
docs = rag.collection.get(include=["documents", "metadatas"])
for doc_id, text, metadata in zip(docs["ids"], docs["documents"], docs["metadatas"]):
    print(f"ID: {doc_id}\nText: {text}\nMetadata: {metadata}\n")
