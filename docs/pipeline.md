# ESG PDF Parsing 파이프라인

## 0. (옵션) PDF 인코딩 보정 (`src/pdf_text_extractor.py`)
- **목적**: PDF 인코딩 문제로 텍스트가 깨지거나 Docling이 읽지 못할 때, **화면 요소(Visual)**를 기반으로 PDF를 재구축(Reconstruction)한다.
- **주요 산출물**: `원본.sanitized.pdf` (MD 파일은 생성하지 않음)
- **실행 예시**
  ```bash
  python3 src/pdf_text_extractor.py --pdf data/input/broken.pdf
  ```
  이후 단계에서는 생성된 `data/input/broken.sanitized.pdf`를 입력으로 사용한다.

## 1. Docling 기반 구조화 (`src/structured_extract.py`)
- **목적**: 페이지별 Markdown, 표(텍스트+JSON), 그림 이미지를 생성하고 `page.json`에 모든 메타데이터를 기록- **주요 산출물** (`data/pages_structured/page_XXXX/`)

- **보고서별 폴더 구조**
  - 기본 출력 경로(`data/pages_structured`)를 사용할 경우 PDF 파일명을 정규화하여 자동으로 하위 폴더를 만든다. 예: `2025_HDEC_Sustainability_Report_K.pdf` → `data/pages_structured/2025_HDEC_Sustainability_Report_K/page_0026/…`
  - 직접 폴더명을 정하고 싶으면 `--report-name my_client`로 지정하면 된다.
  - `--output-dir`를 다른 경로로 설정하면 해당 경로에 그대로 저장되므로, 필요에 따라 기존 평면 구조도 유지 가능.
- **입력**: 원본 PDF 또는 **0번 단계에서 생성된 sanitized PDF**.
- **주요 산출물** (`data/pages_structured/<보고서명>/page_XXXX/`)
  - `page.md` / `page.png`
  - `tables/table_***.(md|json|png)`
  - `figures/figure_***.png`
  - `page.json`: 페이지 번호, 표/그림 bbox, GPT 요약 경로, OCR/비교 결과 등 메타데이터.
- **실행 예시**
  ```bash
  # 원본 사용 시
  python3 src/structured_extract.py --pdf data/input/normal.pdf --pages 25-27

  # 보정된 PDF 사용 시
  python3 src/structured_extract.py --pdf data/input/broken.sanitized.pdf --pages 25-27
  ```
  `OPENAI_API_KEY` 또는 `OPEN_AI_API_KEY` 환경변수를 사용하며, `--gpt-api-key`로 직접 지정할 수도 있다.

## 2. 표 텍스트 추출 (`src/table_ocr.py`)
- **목적**: `tables/table_***.png` 영역의 텍스트를 추출해 `table_***.ocr.json`으로 저장하고, `page.json`의 `ocr_path`/`ocr_preview`를 갱신한다.
- **백엔드 선택**
  - `--backend pymupdf` (기본값): Docling이 기록한 표 bbox를 이용해 PDF 원본에서 직접 텍스트를 읽어온다. PDF에 내장된 글자를 그대로 쓰므로 숫자 정확도가 높다.
  - `--backend rapidocr`: 기존과 동일하게 표 이미지를 RapidOCR 모델에 넣어 OCR한다. 스캔 PDF처럼 텍스트가 없는 경우에 사용.
  - `--pdf`로 PyMuPDF가 읽을 PDF 경로를 지정할 수 있으며, 생략 시 `data/input`의 첫 번째 PDF를 자동으로 사용.
- **실행 예시**
  ```bash
  # PDF 텍스트 기반 추출 (기본)
  python3 src/table_ocr.py --pages 25-27 --backend pymupdf

  # RapidOCR로 이미지 OCR
  python3 src/table_ocr.py --pages 25-27 --backend rapidocr
  ```
  `--overwrite`로 기존 결과를 덮어쓸 수 있다.

## 3. 그림/도식 GPT 설명 (`src/figure_ocr.py`)
- **목적**: GPT-4o-mini에 그림 이미지를 전달해 다이어그램, 화살표 흐름, 강조 텍스트를 서술한 Markdown(`figure_***.desc.md`) 생성.
- `page.json`의 `figures[*].description_path`가 채워지며, `figure_ocr.py`는 다음과 같은 개선 로직을 포함한다.
  - **아이콘 스킵**: 그림 bbox 면적이 페이지 대비 `1%` 미만이거나 헤더 영역(`상단 12%`)에 있으면 자동으로 건너뛰어 비용을 절약한다.
  - **페이지 맥락 주입**: `page.md`에서 앞부분을 잘라 GPT 프롬프트에 포함시켜 그림 설명이 본문 맥락과 연결되도록 한다.
  - **텍스트 없는 사진 스킵(옵션)**: `--skip-textless`를 지정하면 RapidOCR로 텍스트/숫자가 감지되지 않는 데다가 색상·채도 특성상 사진으로 보이는 이미지(인물/건물 등)만 `[SKIP PHOTO]`로 건너뛴다. 순수 다이어그램은 계속 GPT에 전달된다.
- **실행 예시**
  ```bash
  python3 src/figure_ocr.py --pages 25-27 --model gpt-4o-mini
  ```
  실행 전 `OPENAI_API_KEY`(또는 `OPEN_AI_API_KEY`)를 반드시 설정.

## 4. 표 숫자 검증 (`src/table_diff.py`)
- **목적**: Docling 표 JSON과 RapidOCR 결과에서 추출한 숫자를 비교해 차집합을 `table_***.diff.json`으로 저장.
- `page.json`에 `diff_path`, `diff_summary`가 기록되어 검토 대상 숫자를 바로 찾을 수 있다.
- **실행 예시**
  ```bash
  python3 src/table_diff.py --pages 25-27
  ```

## 5. GPT 프롬프트 구성 가이드
## 실행 순서 요약
1. `structured_extract.py` – 페이지 구조화 + (옵션) GPT 요약
2. `table_ocr.py` – 표 이미지 OCR
3. `figure_ocr.py` – 그림/도식 GPT 설명
4. `table_diff.py` – 숫자 비교 및 검증 태깅

필요에 따라 RapidOCR 결과나 그림 설명을 GPT 프롬프트에 주입해 “텍스트·표·이미지”를 모두 커버하는 해석 워크플로를 구성할 수 있다.

> **참고**: 모든 스크립트가 실행 시 `.env` 파일을 자동으로 로드하므로, `OPENAI_API_KEY` 등을 `.env`에 저장해 두면 별도의 `export` 없이 재사용할 수 있다.
