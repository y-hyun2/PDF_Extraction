# 🗄️ Vector DB 설계 (페이지/청크 2단 구조)

이번 파이프라인에서는 페이지 대표 검색과 세부 검색을 나누기 위해 **두 개의 Chroma 컬렉션**을 운용한다.

## 1. 컬렉션 개요

| 컬렉션 | 목적 | 텍스트 구성 | 메타데이터(주요) |
|--------|------|-------------|----------------|
| `esg_pages` | 페이지 단위 대표 검색 | GPT 요약(`gpt-4o-mini`, `temperature=0.3`, `max_output_tokens=800`, `OPENAI_API_KEY` 필수)로 페이지 본문/표/그림 설명과 `page.png` 이미지를 함께 보내 지정 포맷으로 정리 | `doc_id`, `page_id`, `page_no`, `page_image_path`, `table_ids`(JSON 문자열), `figure_ids`(JSON 문자열), `company_name`, `report_year`, `filename`, `created_at` |
| `esg_chunks` | 정밀 검색(본문 청크/표/그림 설명) | - 본문 청크(문자 기준 512/overlap 50)<br>- 표 요약(셀 텍스트 행/열 순 재조합 + diff 정보)<br>- 그림 설명(`figure_***.desc.md`) | `source_type`(`page_text`/`table`/`figure`), `doc_id`, `page_id`, `page_no`, `chunk_index` or `table_id`/`figure_id`, `image_path`, `company_name`, `report_year`, `filename`, `created_at` |

> **주의**: Chroma는 메타데이터 값이 `str/int/float/bool/None`만 허용하므로 리스트(`table_ids`, `figure_ids`)는 JSON 문자열로 저장한다. 조회 시 `json.loads(metadata["table_ids"])` 형태로 다시 리스트로 복원한다.

## 2. 데이터 흐름 요약

1. MySQL에서 `pages`, `doc_tables`, `doc_figures`, `table_cells`를 조회한다.
2. 페이지별로 표/그림을 그룹화하여 대표 텍스트와 세부 청크를 생성한다.
3. SentenceTransformer `BAAI/bge-m3` 모델로 임베딩하고 Chroma에 `upsert`한다.

```
pages ─┬─> page summary (esg_pages)
       ├─> page chunks (esg_chunks, source_type=page_text)
       ├─> tables + table_cells → table summary (source_type=table)
       └─> figures → figure description (source_type=figure)
```

## 3. 실행 스크립트 요약 (`src/build_vector_db.py`)

```bash
python3 src/build_vector_db.py --reset
```

주요 동작:
- Chroma PersistentClient를 `vector_db/` 경로에 생성.
- `--reset` 시 기존 `esg_pages`, `esg_chunks` 컬렉션 삭제 후 재생성.
- 임베딩 모델 `BAAI/bge-m3`는 SentenceTransformer가 첫 실행 시 자동 다운로드.
- 페이지 대표 텍스트는 OpenAI GPT(`gpt-4o-mini`, `OPENAI_API_KEY` 필요)로 전용 프롬프트를 사용해 한글 요약을 생성하고, `page.png` 이미지를 함께 올려 표/그림 내용을 텍스트로 풀어낸다.
- 표 셀 데이터는 페이지 단위로 `fetch_table_cells()`를 호출해 메모리 사용 최소화.
- 각 upsert 배치는 `BATCH_SIZE=32`로 나눠 처리.
- 벡터 검색(`src/search_vector_db.py`)은 기본적으로 `hybrid` 모드로 semantic 후보(개수는 `--semantic-top-k`, 기본 40)를 넓게 뽑고, 그 후보에 대해 BM25 점수를 다시 계산(BM25는 페이지 대표 요약 + 해당 페이지의 본문/표/그림 청크를 모두 합친 텍스트를 corpus로 사용)해 정규화 후 가중합 → 로컬 Reranker(`BAAI/bge-reranker-v2-m3`) 순으로 최종 정렬한다. 최종 출력 시 같은 페이지(`doc_id`+`page_no`)에 해당하는 문서가 여러 개 있으면 하나만 남긴다. `--show-scores`를 주면 semantic/BM25/combined 점수와 reranker 점수를 함께 출력할 수 있다. (키워드 검색을 위해 `kiwipiepy` 설치가 필수)
```
embed_and_upsert(collection, model, ids, documents, metadatas)
```

## 4. 샘플 메타데이터 구조

```json
{
  "source_type": "table",
  "doc_id": 5,
  "page_id": 42,
  "page_no": 12,
  "table_id": 314,
  "table_title": "Scope별 배출량",
  "company_name": "현대건설",
  "report_year": 2023,
  "filename": "2023_HDEC_Report.pdf",
  "image_path": "page_0042/tables/table_001.png",
  "diff_present": true,
  "created_at": "2026-01-27T12:34:56"
}
```

페이지 컬렉션 메타 예시:
```json
{
  "doc_id": 5,
  "page_id": 42,
  "page_no": 12,
  "company_name": "현대건설",
  "report_year": 2023,
  "filename": "2023_HDEC_Report.pdf",
  "page_image_path": "page_0042/page.png",
  "table_ids": "[\"314\", \"315\"]",
  "figure_ids": "[\"789\"]",
  "created_at": "2026-01-27T12:34:56"
}
```


## 5. 실행/추가 팁
- 페이지/청크 컬렉션을 기준으로 `doc_id` → `page_no` → `table_id/figure_id`를 필터링하는 API/서비스 만들기.
- `table_ids`/`figure_ids` JSON 문자열을 역직렬화해 원본 표/그림 데이터를 UI에서 즉시 노출.
- PDF 이미지 썸네일을 외부 스토리지에 두고 `image_path` 대신 URL을 메타데이터로 저장.
- 검색 요청이 많은 경우, reranker 결과를 캐시하거나 외부 검색엔진과 연동해 응답 속도 최적화.
- 특정 페이지의 GPT 요약이 궁금하면 `python3 src/debug_page_summary.py <doc_id> <page_no>`를 실행해 `esg_pages` 컬렉션에 저장된 요약 텍스트와 메타데이터를 확인할 수 있다.

이 설계를 기준으로 `build_vector_db.py`와 `docs/pipeline.md`가 이미 최신화되어 있으니, 추가 요구사항이 생기면 해당 스크립트와 문서를 함께 수정하면 된다.
