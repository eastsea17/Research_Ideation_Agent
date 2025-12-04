from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import re
import json
import config
from operator import itemgetter # 논문 정렬을 위해 추가

# --- Pydantic 데이터 모델 정의 ---
class ResearchTopic(BaseModel):
    title: str = Field(description="The title of the research topic")
    background: str = Field(description="Background context of the research")
    necessity: str = Field(description="Why this research is needed")
    table_of_contents: List[str] = Field(description="Proposed table of contents")
    expected_effects: str = Field(description="Expected effects or impact of the research")
    related_papers: List[Dict[str, Any]] = Field(default=[], description="Top similar papers with full metadata")

class TopicList(BaseModel):
    topics: List[ResearchTopic] = Field(description="List of generated research topics")

# --- 프롬프트에 주입할 예시 JSON (One-Shot Learning) ---
EXAMPLE_JSON_STRUCTURE = """
{
  "topics": [
    {
      "title": "Applying AI to Patent Claim Analysis",
      "background": "Current patent analysis relies heavily on manual expert review...",
      "necessity": "Manual review is time-consuming and prone to human error...",
      "table_of_contents": [
        "1. Introduction to Patent Claims",
        "2. NLP Techniques for Legal Text",
        "3. System Architecture",
        "4. Evaluation Metrics"
      ],
      "expected_effects": "Reduce analysis time by 50% and increase accuracy..."
    }
  ]
}
"""

class TopicGenerator:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.MODEL_GENERATOR,
            temperature=config.GENERATOR_TEMPERATURE,
            keep_alive="5m"
        )

    def generate_topics(self, vector_db, keyword, num_topics=config.DEFAULT_TOPIC_COUNT):
        print(f"Generating topics for keyword: {keyword}...")
        
        # 1. RAG: 관련 논문 검색 및 최신 논문 추출 로직 추가
        try:
            retriever = vector_db.as_retriever(search_kwargs={"k": config.VECTOR_DB_SEARCH_K})
            docs = retriever.invoke(keyword)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 모든 문서의 메타데이터를 사용하여 최신 논문 5개 추출
            all_docs_metadata = [doc.metadata for doc in docs]
            
            # publication_year를 기준으로 내림차순 정렬
            # itemgetter를 사용하여 year가 없는 경우 (None)를 0으로 처리하여 정렬 오류 방지
            sorted_docs = sorted(
                all_docs_metadata, 
                key=lambda x: x.get('year', 0), 
                reverse=True
            )
            
            # 최신 5개 논문의 제목만 추출
            latest_titles = [
                f"- [{doc.get('year', 'N/A')}] {doc.get('title', 'Unknown Title')}"
                for doc in sorted_docs[:5]
            ]
            
            latest_papers = "\n".join(latest_titles)
            
        except Exception as e:
            print(f"⚠️ Warning: Context or Latest Paper retrieval failed ({e}). Proceeding without specific latest titles.")
            context = "No specific context provided."
            latest_papers = "No latest papers found."
        
# 2. 프롬프트 작성 (심층 연구 프롬프트 적용)
        prompt_template = """
        You are a Senior Principal Investigator (PI) at a top-tier research institute.
        Your goal is to propose {num_topics} groundbreaking research agendas related to "{keyword}" that could be published in top-tier journals (e.g., Nature, Science, AAAI, NeurIPS).

        --- 1. STATE OF THE ART (SOTA) ANALYSIS ---
        The following titles represent the most recent developments (latest papers). 
        Analyze them to understand the current research frontier:
        {latest_papers}
        
        --- 2. KNOWLEDGE BASE (RAG Context) ---
        Use these specific details to ground your proposals in reality and technical feasibility:
        {context}
        
        --- 3. IDEATION FRAMEWORK (Chain of Thought) ---
        To generate the topics, you MUST first engage in a deep reasoning process using the <think> tag.
        Inside the <think> block, follow this "Critic -> Solution" logic:
        
        <think>
        1. CRITIC (Identify Limitations):
           - Critically analyze the provided "Latest Papers" and "Context".
           - Explicitly state what is MISSING, FLAWED, or OUTDATED in the current research.
           - Why are existing approaches insufficient? (e.g., "Current methods rely on X which is computationally expensive," or "They fail to address Y scenario").
           
        2. SOLUTION (Propose Alternatives):
           - For each limitation identified, propose a specific, novel alternative.
           - How can we overcome the identified flaws? (e.g., "Instead of X, we can use Z to reduce complexity," or "Integrate A and B to solve Y").
           - Verify if this solution is truly "disruptive" and not just an incremental improvement.
        </think>
        
        --- 4. STRICT CONSTRAINTS ---
        - Avoid "incremental" improvements (e.g., "Using X for Y"). Focus on "disruptive" ideas.
        - Ensure "Necessity" clearly argues why current methods fail (based on your <think> analysis).
        - Ensure "Expected Effects" includes quantitative or specific qualitative breakthroughs (e.g., "Reducing complexity from O(N^2) to O(N)").
        
        --- 5. OUTPUT FORMAT ---
        Provide ONLY the JSON object. Do NOT include markdown formatting (```json), explanations, or schema definitions ($defs).
        The <think> block should come BEFORE the JSON output, but the final response should be parsed to extract the JSON.
        (Note: The system will extract the JSON, so you can output the <think> block first, followed by the JSON).
        
        Follow the structure of the EXAMPLE below exactly.

        EXAMPLE JSON OUTPUT:
        {example_json}

        YOUR PROPOSAL:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["num_topics", "keyword", "context", "example_json", "latest_papers"]
        )
        
        # 3. 체인 실행
        chain = prompt | self.llm
        
        try:
            print("⏳ Asking LLM to generate ideas (this may take a moment with local llm)...")
            response_msg = chain.invoke({
                "num_topics": num_topics,
                "keyword": keyword,
                "context": context,
                "example_json": EXAMPLE_JSON_STRUCTURE,
                "latest_papers": latest_papers # 새로운 변수 추가
            })
            
            raw_content = response_msg.content
            
            # 4. 후처리 (Parsing 전 데이터 클리닝)
            if '<think>' in raw_content:
                raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            raw_content = raw_content.replace('```json', '').replace('```', '').strip()
            
            # 5. JSON 파싱
            try:
                json_data = json.loads(raw_content)
                result = TopicList(**json_data)
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parsing Error: {e}")
                print(f"Debug - Raw Content First 500 chars: {raw_content[:500]}")
                return []
            except Exception as e:
                print(f"❌ Validation Error: {e}")
                return []

            print(f"✅ Generated {len(result.topics)} topics successfully.")
            
            # 6. 관련 논문 매핑 (후처리)
            print("Mapping related papers...")
            # (기존 매핑 로직은 동일)
            for topic in result.topics:
                try:
                    similar_docs = vector_db.similarity_search(topic.title, k=1)
                    topic.related_papers = []
                    for doc in similar_docs:
                        topic.related_papers.append({
                            "title": doc.metadata.get("title", "Unknown Title"),
                            "authors": doc.metadata.get("authors", "Unknown Authors"),
                            "year": doc.metadata.get("year", "N/A"),
                            "url": doc.metadata.get("url", "")
                        })
                except Exception as e:
                    print(f"⚠️ Failed to map papers for topic '{topic.title}': {e}")
                    topic.related_papers = []
            
            return result.topics

        except Exception as e:
            print(f"❌ Critical Error in generate_topics: {e}")
            return []

if __name__ == "__main__":
    pass