"""
RAG Answer Generator using Multimodal LLM (Gemma-3n-E4B-it).

Logic:
1. Search Vector DB for query.
2. Retrieve Top-K chunks (Markdown + Metadata).
3. Fetch corresponding Page Images.
4. Construct Multimodal Prompt (Text + Images).
5. Generate Answer.

Handling "All different pages":
- We prioritize unique pages from the Top-K results.
- We limit to N images (e.g., 3) to prevent context overflow.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# 허깅페이스 비공개 모델 접근 토큰을 환경변수에서 불러온다.
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 동일 디렉터리의 검색 모듈을 불러와 중복 제거된 결과를 재활용한다.
sys.path.append(str(Path(__file__).parent))
from search_vector_db import search_vector_db

# 다른 스크립트에서도 재사용할 수 있도록 남겨둔 보조 로더 함수.
def load_model(model_id: str):
    print(f"📦 Loading Model '{model_id}'...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("✅ Model Loaded.")
        return processor, model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

# 메타데이터 힌트를 이용해 data/pages_structured 내 페이지 이미지를 찾는다.
def get_page_image_path(metadata: Dict, page_no: Optional[int]) -> Optional[Path]:
    """Locate the PNG corresponding to a page by inferring the report folder name."""

    # 필수 정보가 없으면 바로 종료한다.
    if page_no is None:
        return None

    base_dir = Path("data/pages_structured")
    if not base_dir.exists():
        return None

    company = metadata.get('company_name') or metadata.get('company') or ''
    report_year = metadata.get('report_year') or metadata.get('year') or ''
    filename = metadata.get('filename') or ''
    direct_report_dir = (
        metadata.get('report_dir')
        or metadata.get('doc_dir')
        or metadata.get('doc_name_hint')
    )

    candidate_dirs: List[str] = []
    seen = set()

    def add_candidate(name: Optional[str]):
        if not name:
            return
        clean = str(name).strip()
        if not clean or clean in seen:
            return
        candidate_dirs.append(clean)
        seen.add(clean)

    # 메타데이터에 폴더명이 명시돼 있으면 최우선으로 시도한다.
    add_candidate(direct_report_dir)

    if filename:
        stem = Path(filename).stem
        add_candidate(stem)
        add_candidate(stem.replace(" ", "_"))
        add_candidate(stem.replace("-", "_"))

    if company and report_year:
        combos = [
            f"{report_year}_{company}_Report",
            f"{company}_{report_year}_Report",
            f"{report_year}_{company}",
            f"{company}_{report_year}",
        ]
        for combo in combos:
            add_candidate(combo)
            add_candidate(combo.replace(" ", "_"))

    # 위에서 수집한 후보 디렉터리를 순차적으로 검사한다.
    page_dir_name = f"page_{page_no:04d}"
    for candidate in candidate_dirs:
        page_path = base_dir / candidate / page_dir_name / "page.png"
        if page_path.exists():
            return page_path

    # 보조 수단: 회사/연도 키워드를 모두 포함한 폴더를 훑는다.
    company_upper = company.upper()
    year_str = str(report_year)
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
        folder_name_upper = folder.name.upper()
        if company_upper and company_upper not in folder_name_upper:
            continue
        if year_str and year_str not in folder.name:
            continue
        candidate_path = folder / page_dir_name / "page.png"
        if candidate_path.exists():
            return candidate_path

    # 최종 수단: 페이지 폴더가 존재하는 모든 경로를 확인한다.
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
        candidate_path = folder / page_dir_name / "page.png"
        if candidate_path.exists():
            return candidate_path

    return None

# 전체 RAG + VLM 파이프라인을 조립하는 엔트리포인트.
def main():
    # CLI 인자를 정의해 질의/필터 방식을 제어한다.
    parser = argparse.ArgumentParser(description="RAG Answer Generator")
    parser.add_argument("query", type=str, help="Question to ask")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Model ID") 
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    
    # 회사/연도 필터 인자
    parser.add_argument("--company", type=str, default=None, help="Filter by Company Name (e.g. HDEC)")
    parser.add_argument("--year", type=int, default=None, help="Filter by Report Year (e.g. 2023)")
    
    args = parser.parse_args()
    
    # 1단계: 가장 관련 있는 페이지를 벡터 DB에서 검색한다.
    print(f"🔎 Searching for: '{args.query}'")
    if args.company or args.year:
        print(f"   Filters: Company='{args.company}', Year='{args.year}'")

    results = search_vector_db(args.query, top_k=args.top_k)
    
    # 필요 시 회사/연도 조건으로 결과를 한 번 더 필터링한다.
    company_filter = args.company.lower() if args.company else None
    year_filter = str(args.year) if args.year else None

    if company_filter or year_filter:
        filtered = []
        for res in results:
            meta = res.get('metadata', {})
            company_name = str(meta.get('company_name') or meta.get('company') or '').lower()
            report_year = str(meta.get('report_year') or meta.get('year') or '')

            if company_filter and company_filter not in company_name:
                continue
            if year_filter and year_filter != report_year:
                continue
            filtered.append(res)

        if not filtered:
            print("⚠️ 필터 조건에 맞는 결과가 없어 전체 결과를 사용합니다.")
        else:
            results = filtered

    if not results:
        print("Test ended: No results found.")
        return

    # 2단계: 페이지 단위로 텍스트/이미지를 묶어 동기화한다.
    unique_pages = {}  # page_key -> {image_path, texts, page_info} 구조

    for res in results:
        meta = res.get('metadata', {})
        doc_year = meta.get('report_year', 'UnknownYear')
        company = meta.get('company_name', 'UnknownCompany')
        page_no = meta.get('page_no')
        chunk_text = res.get('content', '')

        if page_no is None:
            continue

        page_key = f"{company}_{doc_year}_{page_no}"

        if page_key not in unique_pages:
            doc_name_hint = f"{doc_year}_{company}_Report"
            meta_with_hint = dict(meta)
            meta_with_hint.setdefault('doc_name_hint', doc_name_hint)
            # 가능한 한 정확한 페이지 이미지 경로를 붙인다.
            img_path = get_page_image_path(meta_with_hint, page_no)

            unique_pages[page_key] = {
                "image_path": img_path,
                "texts": [],
                "page_info": f"{company} {doc_year} Sustainability Report (Page {page_no})"
            }

        if chunk_text:
            unique_pages[page_key]["texts"].append(chunk_text)

    # 3단계: 멀티모달 지시형 모델과 프로세서를 로드한다.
    print(f"📦 Loading Model '{args.model}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # 더 안정적인 추론을 위해 bfloat16 사용
            trust_remote_code=True,
            token=HF_TOKEN 
        ).eval()
        processor = AutoProcessor.from_pretrained(
            args.model, 
            trust_remote_code=True,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if "gated" in str(e).lower() or "401" in str(e):
             print("💡 Tip: Ensure HF_TOKEN is set in .env and you have access to the model.")
        return

    # 4단계: 이미지와 텍스트가 섞인 멀티모달 대화 프롬프트를 구성한다.
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an ESG report analyst. Use only the provided context. "
                "Never hallucinate or fabricate data. Cite page-level evidence explicitly. "
                "When quoting tables or figures, copy the numbers exactly as shown."
            ),
        },
    ]
    
    # 사용자 메시지에 텍스트/이미지를 번갈아 배치한다.
    user_content = []
    user_content.append({"type": "text", "text": f"Question: {args.query}\n\nContexts:\n"})
    
    images_loaded = []
    
    # VRAM 절약을 위해 상위 3개 페이지까지만 사용한다.
    for i, (key, data) in enumerate(list(unique_pages.items())[:3]):
        if data["image_path"]:
            user_content.append({"type": "text", "text": f"--- Page Image ({data['page_info']}) ---\n"})
            user_content.append({"type": "image", "image": str(data["image_path"])}) # Processor가 경로 문자열 또는 PIL 이미지를 처리
            images_loaded.append(data["image_path"])
        
        texts_combined = "\n... \n".join(data["texts"])
        user_content.append({"type": "text", "text": f"\n[Extracted Text for {data['page_info']}]:\n{texts_combined}\n\n"})

    user_content.append({"type": "text", "text": "Answer:"})
    messages.append({"role": "user", "content": user_content})

    # 5단계: 전체 컨텍스트를 조건으로 답변을 생성한다.
    print("🤖 Generating Answer...")
    
    # 모델 입력 텐서를 준비한다.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # PIL 이미지 객체를 불러온다.
    pil_images = [Image.open(p) for p in images_loaded] if images_loaded else None
    
    inputs = processor(
        text=[text],
        images=pil_images,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "="*40)
    print("📝 Answer:")
    print("="*40)
    print(output_text)
    print("="*40)

if __name__ == "__main__":
    main()
