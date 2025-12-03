import argparse
import sys
import os
import requests
from agents.collector import PaperCollector
from agents.generator import TopicGenerator
from agents.evaluator import TopicEvaluator
from utils.report_generator import generate_html_report
import config

def unload_model(model_name):
    """
    Ollama API를 통해 특정 모델을 메모리에서 즉시 언로드합니다.
    (M4 Pro 24GB 환경에서 임베딩 모델과 생성 모델 충돌 방지용)
    """
    if not model_name:
        return
        
    try:
        print(f"🧹 Requesting unload for model: {model_name}...")
        requests.post(f"{config.OLLAMA_BASE_URL}/api/generate", json={
            "model": model_name,
            "keep_alive": 0
        })
        print(f"✅ Successfully unloaded: {model_name}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to unload model {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Topic Brainstorming Tool")
    parser.add_argument("keyword", help="Seed keyword for research topic brainstorming")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_PAPER_LIMIT, help=f"Number of papers to fetch (default: {config.DEFAULT_PAPER_LIMIT})")
    parser.add_argument("--topics", type=int, default=config.DEFAULT_TOPIC_COUNT, help=f"Number of topics to generate (default: {config.DEFAULT_TOPIC_COUNT})")
    
    args = parser.parse_args()
    keyword = args.keyword
    
    print(f"🚀 Starting Brainstorming Session for: '{keyword}'")
    
    # ---------------------------------------------------------
    # 1. Collect Papers & Vector DB
    # ---------------------------------------------------------
    print("\n--- Step 1: Collecting Papers & Creating Vector DB ---")
    collector = PaperCollector()
    papers = collector.fetch_papers(keyword, limit=args.limit)
    
    if not papers:
        print("❌ No papers found. Exiting.")
        sys.exit(1)
        
    vector_db = collector.create_vector_db(papers)
    
    if vector_db is None:
        print("❌ Failed to create Vector DB. Exiting.")
        sys.exit(1)

    # [최적화 1] 임베딩 모델 메모리 해제
    print("Optimization: Unloading embedding model...")
    unload_model(config.MODEL_EMBEDDING)
    
    # ---------------------------------------------------------
    # 2. Generate Topics
    # ---------------------------------------------------------
    print("\n--- Step 2: Generating Research Topics ---")
    generator = TopicGenerator()
    topics = generator.generate_topics(vector_db, keyword, num_topics=args.topics)
    
    if not topics:
        print("❌ Failed to generate topics. Exiting.")
        sys.exit(1)

    # [최적화 2] Generator 모델 해제 (Evaluator와 다른 모델을 쓸 경우를 대비해 미리 해제 가능)
    # 하지만 보통 Generator(R1) -> Evaluator(GPT-OSS) 순서라면, 
    # R1은 무거우므로 여기서 바로 내리는 것이 좋습니다.
    if config.MODEL_GENERATOR != config.MODEL_EVALUATOR:
        print("Optimization: Unloading generator model...")
        unload_model(config.MODEL_GENERATOR)
        
    # ---------------------------------------------------------
    # 3. Evaluate Topics
    # ---------------------------------------------------------
    print("\n--- Step 3: Evaluating Topics ---")
    evaluator = TopicEvaluator()
    evaluated_topics = evaluator.evaluate_topics(topics)
    
    # ---------------------------------------------------------
    # 4. Generate Report (English)
    # ---------------------------------------------------------
    print("\n--- Step 4: Generating Report (English) ---")
    os.makedirs(config.OUTPUT_REPORT_DIR, exist_ok=True)
    filename_en = os.path.join(config.OUTPUT_REPORT_DIR, f"report_{keyword.replace(' ', '_')}.html")
    generate_html_report(evaluated_topics, filename=filename_en)
    print(f"📄 English Report saved to: {filename_en}")

    # ---------------------------------------------------------
    # 5. Translate and Generate Report (Korean) & Cleanup
    # ---------------------------------------------------------
    print("\n--- Step 5: Translating and Generating Report (Korean) ---")
    try:
        from agents.translator import TopicTranslator
        translator = TopicTranslator()
        translated_topics = translator.translate_topics(evaluated_topics, target_language="Korean")
        
        filename_ko = os.path.join(config.OUTPUT_REPORT_DIR, f"report_{keyword.replace(' ', '_')}_ko.html")
        generate_html_report(translated_topics, filename=filename_ko)
        print(f"🇰🇷 Korean Report saved to: {filename_ko}")

    except Exception as e:
        print(f"⚠️ Translation failed or skipped: {e}")
    
    finally:
        # ---------------------------------------------------------
        # [최적화 3] Final Cleanup: 모든 모델 언로드
        # ---------------------------------------------------------
        print("\n--- Final Cleanup: Unloading All Models ---")
        
        # Generator가 위에서 안 내려갔을 수도 있으니 확실히 다시 시도
        unload_model(config.MODEL_GENERATOR)
        
        # Evaluator 모델 언로드
        unload_model(config.MODEL_EVALUATOR)
        
        # Translator 모델이 별도로 있다면 언로드 (보통 Evaluator와 같은 모델을 쓰더라도 안전하게 호출)
        # config에 TRANSLATOR 모델이 정의되어 있다고 가정, 없다면 EVALUATOR 사용
        model_translator = getattr(config, 'MODEL_TRANSLATOR', config.MODEL_EVALUATOR)
        if model_translator != config.MODEL_GENERATOR and model_translator != config.MODEL_EVALUATOR:
             unload_model(model_translator)

    print(f"\n✅ Success! All tasks completed and memory cleared.")

if __name__ == "__main__":
    main()