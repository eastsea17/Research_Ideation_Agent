import requests
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import config
import csv
import os
from datetime import datetime
import time

class PaperCollector:
    def __init__(self):
        # [수정] timeout 인자 제거 (Pydantic ValidationError 해결)
        self.embeddings = OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.MODEL_EMBEDDING
        )
        self.vector_db = None
        # Config에 없는 경우를 대비한 기본값
        self.AUTHORS_LIMIT = getattr(config, 'AUTHORS_LIMIT', 3)
        self.INSTITUTIONS_LIMIT = getattr(config, 'INSTITUTIONS_LIMIT', 3)

    def fetch_papers(self, keyword, limit=config.DEFAULT_PAPER_LIMIT):
        print(f"Fetching papers for keyword: {keyword}...")
        url = config.OPENALEX_API_URL
        params = {
            "search": keyword,
            "per-page": limit,
            "filter": "has_abstract:true"
        }
        headers = {
            "User-Agent": f"mailto:{config.USER_AGENT_EMAIL}"
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            
            papers = []
            for result in results:
                title = result.get("title")
                abstract_inverted = result.get("abstract_inverted_index")
                abstract = self._reconstruct_abstract(abstract_inverted)
                
                authorships = result.get("authorships", [])
                authors = []
                institutions = []
                
                for authorship in authorships[:self.AUTHORS_LIMIT]:
                    author = authorship.get("author", {})
                    author_name = author.get("display_name")
                    if author_name:
                        authors.append(author_name)
                    for inst in authorship.get("institutions", []):
                        inst_name = inst.get("display_name")
                        if inst_name and inst_name not in institutions:
                            institutions.append(inst_name)
                
                institutions = institutions[:self.INSTITUTIONS_LIMIT]
                
                if title and abstract:
                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "url": result.get("id"),
                        "publication_year": result.get("publication_year"),
                        "authors": authors, 
                        "institutions": institutions
                    })
            
            print(f"Found {len(papers)} papers.")
            self.save_papers_to_csv(papers, keyword)
            return papers
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []

    def save_papers_to_csv(self, papers, keyword):
        if not papers: return
        os.makedirs(config.OUTPUT_CSV_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"papers_{keyword.replace(' ', '_')}_{timestamp}.csv"
        filepath = os.path.join(config.OUTPUT_CSV_DIR, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['title', 'abstract', 'url', 'publication_year', 'authors', 'institutions']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for paper in papers:
                    row = paper.copy()
                    row['authors'] = '; '.join(paper.get('authors', []))
                    row['institutions'] = '; '.join(paper.get('institutions', []))
                    writer.writerow(row)
            print(f"Papers saved to: {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def _reconstruct_abstract(self, inverted_index):
        if not inverted_index: return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions: word_positions.append((pos, word))
        word_positions.sort()
        return " ".join([word for _, word in word_positions])

    def create_vector_db(self, papers):
        if not papers:
            print("No papers to index.")
            return None

        print("Creating Vector DB...")
        documents = []
        
        # 메타데이터 전처리: List -> String 변환
        for paper in papers:
            # [수정 1] 텍스트 길이 안전 장치 (Abstract가 너무 길면 Ollama가 뻗을 수 있음)
            raw_abstract = paper['abstract']
            if len(raw_abstract) > 1000: # 1000자 제한 (약 500~700 토큰)
                raw_abstract = raw_abstract[:1000] + "...(truncated)"
                
            content = f"Title: {paper['title']}\nAbstract: {raw_abstract}"
            
            metadata = {
                "title": paper['title'],
                "url": paper['url'],
                "year": paper['publication_year'],
                "authors": ", ".join(paper.get('authors', [])),
                "institutions": ", ".join(paper.get('institutions', []))
            }
            documents.append(Document(page_content=content, metadata=metadata))

        print(f"Initializing Vector DB with {len(documents)} documents...")
        
        # DB 초기화
        self.vector_db = Chroma(
            embedding_function=self.embeddings,
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        
        # 안정적인 배치 처리 및 재시도 로직
        total_docs = len(documents)
        batch_size = config.VECTOR_DB_BATCH_SIZE
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch)} docs)...")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.vector_db.add_documents(documents=batch)
                    break # 성공 시 다음 배치로
                except Exception as e:
                    print(f"⚠️ Error adding batch {i} (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5) # Ollama 서버 숨고르기
                    else:
                        print(f"❌ Failed batch {i} after retries. Skipping.")
        
        print("✅ Vector DB creation completed.")
        return self.vector_db

    def query_db(self, query, k=config.VECTOR_DB_SEARCH_K):
        # vector_db가 없으면 로드 시도
        if not self.vector_db:
             self.vector_db = Chroma(
                embedding_function=self.embeddings,
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.CHROMA_PERSIST_DIRECTORY
            )
        return self.vector_db.similarity_search(query, k=k)