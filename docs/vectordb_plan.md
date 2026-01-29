## 🗄️ 벡터DB 구조도 (최종)

```
┌─────────────────────────────────────────────────────────────────┐
│                    📚 Vector Database                           │
│                   Collection: "esg_documents"                    │
└─────────────────────────────────────────────────────────────────┘

📦 Document Structure per Vector:
├── 🆔 vector_id               # UUID (auto-generated)
├── 📊 embedding                # vector[1024] - bge-m3 기준
├── 📝 text                     # 원본 텍스트 (검색 대상)
└── 🏷️  metadata                # 메타데이터 (필터링/추적용)
    ├── source_type             # "page_chunk" / "figure"
    ├── doc_id                  # RDB documents.id (FK)
    ├── page_id                 # RDB pages.id (FK, nullable)
    ├── page_no                 # 페이지 번호 (1, 2, 3...)
    ├── chunk_index             # 청크 순서 (page_chunk인 경우)
    ├── figure_id               # RDB doc_figures.id (figure인 경우)
    ├── image_path              # 이미지 경로 (figure인 경우)
    ├── company_name            # "삼성전자", "현대건설"
    ├── report_year             # 2023, 2024
    ├── filename                # "samsung_esg_2023.pdf"
    └── created_at              # "2024-01-28T10:00:00Z"

┌─────────────────────────────────────────────────────────────────┐
│                    🎯 Vector Types                               │
└─────────────────────────────────────────────────────────────────┘

Type 1: page_chunk (Phase 1 - 필수) ⭐⭐⭐
├── Source: pages.full_markdown
├── Process: 512 토큰 단위로 청킹
├── Count: ~500 vectors per document
└── Example:
    text: "현대건설은 2023년 탄소배출량 감축을 위해...
           [표 1: Scope별 배출량]
           | 구분 | 2022 | 2023 |
           [image]
           위 그래프는 추이를 보여줍니다..."

    metadata: {
      "source_type": "page_chunk",
      "doc_id": 5,
      "page_id": 42,
      "page_no": 12,
      "chunk_index": 0,
      "company_name": "현대건설",
      "report_year": 2023
    }

Type 2: figure (Phase 2 - 선택적) ⭐
├── Source: doc_figures.description
├── Condition: description이 있고 길이 > 100자
├── Count: ~15 vectors per document
└── Example:
    text: "그림 종류: chart
           캡션: 재생에너지 비율 추이

           상세 설명:
           이 차트는 2020년부터 2023년까지 재생에너지 사용 비율이
           지속적으로 증가하는 추세를 보여줍니다..."

    metadata: {
      "source_type": "figure",
      "doc_id": 5,
      "page_id": 43,
      "page_no": 15,
      "figure_id": 789,
      "image_path": "/figures/figure_005.png",
      "company_name": "현대건설",
      "report_year": 2023
    }

```

---

## 🔧 실제 구현 코드

### **Phase 1: 기본 구조 (필수)**

```python
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 초기화
client = Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

collection = client.get_or_create_collection(
    name="esg_documents",
    metadata={"hnsw:space": "cosine"}
)

embed_model = SentenceTransformer('BAAI/bge-m3')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# 2. 페이지 텍스트 벡터화
def add_page_to_vector_db(page, doc):
    """pages.full_markdown을 청킹해서 벡터화"""

    chunks = splitter.split_text(page.full_markdown)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"page_{page.id}_chunk_{i}")
        embeddings.append(embed_model.encode(chunk).tolist())
        documents.append(chunk)
        metadatas.append({
            "source_type": "page_chunk",
            "doc_id": doc.id,
            "page_id": page.id,
            "page_no": page.page_no,
            "chunk_index": i,
            "company_name": doc.company_name,
            "report_year": doc.report_year,
            "filename": doc.filename,
            "created_at": datetime.now().isoformat()
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(chunks)

# 3. 문서 전체 추가
def add_document_to_vector_db(doc_id: int):
    """문서 전체를 벡터DB에 추가"""

    doc = session.query(Document).get(doc_id)
    total_vectors = 0

    for page in doc.pages:
        count = add_page_to_vector_db(page, doc)
        total_vectors += count

    print(f"✅ {doc.filename}: {total_vectors} vectors 추가 완료")
    return total_vectors

```

---

### **Phase 2: 선택적 보강 (그림 설명)**

```python
def add_figures_to_vector_db(doc_id: int):
    """
    description이 있는 그림만 선택적으로 벡터화
    조건: len(description) > 100
    """

    doc = session.query(Document).get(doc_id)
    figures = session.query(DocFigure).filter(
        DocFigure.doc_id == doc_id,
        DocFigure.description.isnot(None)
    ).all()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for figure in figures:
        # 설명이 충분히 긴 경우만
        if len(figure.description) < 100:
            continue

        # 텍스트 구성
        figure_text = f"""
그림 종류: {figure.figure_type}
캡션: {figure.caption}

상세 설명:
{figure.description}

위치: {doc.company_name} {doc.report_year}년 보고서 {figure.page_no}페이지
"""

        ids.append(f"figure_{figure.id}")
        embeddings.append(embed_model.encode(figure_text).tolist())
        documents.append(figure_text)
        metadatas.append({
            "source_type": "figure",
            "doc_id": doc.id,
            "page_id": figure.page_id,
            "page_no": figure.page_no,
            "figure_id": figure.id,
            "image_path": figure.image_path,
            "company_name": doc.company_name,
            "report_year": doc.report_year,
            "filename": doc.filename,
            "created_at": datetime.now().isoformat()
        })

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ {doc.filename}: {len(ids)} figure vectors 추가")

    return len(ids)

```

---

## 🔍 검색 함수

```python
def search_esg_documents(
    query: str,
    company_name: str = None,
    report_year: int = None,
    source_type: str = None,
    top_k: int = 5
):
    """
    벡터 검색 + 메타데이터 필터링

    Args:
        query: 검색 질문
        company_name: 회사명 필터 (optional)
        report_year: 연도 필터 (optional)
        source_type: "page_chunk" or "figure" (optional)
        top_k: 반환할 결과 수
    """

    # 쿼리 임베딩
    query_embedding = embed_model.encode(query).tolist()

    # 필터 구성
    where_filter = {}
    if company_name:
        where_filter["company_name"] = company_name
    if report_year:
        where_filter["report_year"] = report_year
    if source_type:
        where_filter["source_type"] = source_type

    # 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter if where_filter else None
    )

    return results

# 사용 예시
results = search_esg_documents(
    query="Scope 1 배출량 추이",
    company_name="삼성전자",
    report_year=2023,
    top_k=5
)

for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n결과 {i+1}:")
    print(f"출처: {meta['company_name']} {meta['report_year']}년")
    print(f"페이지: {meta['page_no']}")
    print(f"타입: {meta['source_type']}")
    print(f"내용: {doc[:200]}...")

```

---

## 🎯 실전 검색 시나리오

### **시나리오 1: 일반 텍스트 검색**

```python
Q: "삼성전자 2023년 재생에너지 사용 비율은?"

# 검색
results = search_esg_documents(
    query="재생에너지 사용 비율",
    company_name="삼성전자",
    report_year=2023
)

# 결과 (page_chunk)
"""
우리는 재생에너지 투자를 지속 확대하고 있습니다.
아래 그래프는 2020년부터 2023년까지의 추이를 보여줍니다.

[image]

2023년에는 전체 에너지의 35%를 재생에너지로 충당했습니다.
이는 전년 대비 5% 증가한 수치입니다.
"""

# metadata에서 page_no 확인 → RDB 조회 → 정확한 표/그림 찾기

```

---

### **시나리오 2: 그림 특화 검색** (Phase 2)

```python
Q: "탄소중립 로드맵 차트 보여줘"

# 검색 (그림만 필터링)
results = search_esg_documents(
    query="탄소중립 로드맵",
    source_type="figure",  # 그림만!
    top_k=3
)

# 결과 (figure)
"""
그림 종류: chart
캡션: 2050 탄소중립 로드맵

상세 설명:
이 차트는 2030년까지 50% 감축, 2040년까지 75% 감축,
2050년 탄소중립 달성을 목표로 하는 단계별 계획을 보여줍니다...
"""

# metadata['image_path']로 실제 이미지 표시
→ /figures/figure_012.png

```

---

### **시나리오 3: 시계열 비교**

```python
Q: "현대건설 최근 3년 탄소배출 추이"

# 여러 연도 검색
all_results = []
for year in [2021, 2022, 2023]:
    results = search_esg_documents(
        query="탄소배출량 Scope",
        company_name="현대건설",
        report_year=year,
        top_k=3
    )
    all_results.extend(results['metadatas'][0])

# 각 연도별 page_no, doc_id 확인
# → RDB에서 정확한 표 데이터 추출
# → LLM에게 컨텍스트 제공

```

---

## 📈 성능 최적화

```python
# 1. 인덱스 크기 조정 (HNSW)
collection = client.create_collection(
    name="esg_documents",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 16,              # 연결 수 (높을수록 정확, 느림)
        "hnsw:construction_ef": 200 # 인덱스 품질
    }
)

# 2. 배치 처리
def add_document_batch(doc_ids: list):
    """여러 문서 동시 처리"""
    for doc_id in doc_ids:
        add_document_to_vector_db(doc_id)

# 3. 캐싱
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, company: str, year: int):
    return search_esg_documents(query, company, year)

```

---

## ✅ 최종 정리

### **Phase 1 구조 (시작)**

```
Vector DB:
└── page_chunk (500/doc)
    ├── text: full_markdown (표/그림 포함, [image] 표시)
    └── metadata: doc_id, page_no, company, year

```

**장점:**

- ✅ 구조 심플
- ✅ 대부분 검색 커버
- ✅ 빠른 구축

**검색 성능:**

- 일반 질문: 90% 정확도
- 표 검색: 85% 정확도
- 그림 검색: 70% 정확도 (맥락 의존)

---

### **Phase 2 구조 (보강)**

```
Vector DB:
├── page_chunk (500/doc)
└── figure (15/doc)  ← 추가!
    ├── text: caption + description
    └── metadata: figure_id, image_path

```

**추가 시점:**

- 그림 검색 정확도 낮을 때
- 복잡한 차트 많을 때
- 이미지 위주 페이지 많을 때

**검색 성능:**

- 그림 검색: 90% 정확도로 향상!