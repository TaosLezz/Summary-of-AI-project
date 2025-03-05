from rag.core import RAG
import pandas as pd

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

# # Lấy danh sách tất cả các documents trong collection
# docs = rag.collection.get(include=["documents", "metadatas"])
# for doc_id, text, metadata in zip(docs["ids"], docs["documents"], docs["metadatas"]):
#     print(f"ID: {doc_id}\nText: {text}\nMetadata: {metadata}\n")

file_path = r"D:\Hieudev_LLMs\retrieval-backend-with-rag\data\AIDebug_data_clean.csv"
dataframe = pd.read_csv(file_path)
# print(dataframe)
for _, row in dataframe.iterrows():
    doc_id = row["SN"]  # Dùng Serial Number làm ID duy nhất
    text = f"error code: {row['ERROR_CODE']}, Symptom: {row['FAILURE_SYMPTOM']} at location {row['LOCATION']}, station: {row['STATION']}"
    metadata = {
        "customer": row["CUSTOMER"],
        "series": row["SERIES_NAME"],
        "skuno": row["SKUNO"],
        "station": row["STATION"],
        "error_code": row["ERROR_CODE"],
        "location": row["LOCATION"],
        "failure_symptom": row["FAILURE_SYMPTOM"],
        "root_cause": row["ROOT_CAUSE_DESC"],
        "solution": row["ACTIONDESC"],
        # "fail_time": str(row["FAIL_TIME"]),
        # "checkin_time": str(row["CHECKIN_TIME"]),
        # "repair_time": str(row["REPAIR_TIME"]),
        # "repair_count": row["REPARIR_REWORK_COUNT"]
    }
    print(text)
    rag.add_document(doc_id=doc_id, text=text, metadata=metadata)
    # embedding = get_embedding(text)  # Tạo embedding từ mô tả lỗi

    # collection.add(
    #     ids=[doc_id],
    #     embeddings=[embedding],
    #     metadatas=[metadata]
    # )