from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import config
from agents.generator import ResearchTopic
from agents.evaluator import EvaluatedTopic, EvaluationResult

class TranslatedContent(BaseModel):
    title: str = Field(description="Translated title")
    background: str = Field(description="Translated background")
    necessity: str = Field(description="Translated necessity")
    table_of_contents: List[str] = Field(description="Translated table of contents")
    expected_effects: str = Field(description="Translated expected effects")
    reasoning: str = Field(description="Translated evaluation reasoning")

class TopicTranslator:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.MODEL_EVALUATOR, # Using gpt-oss:20b for translation as well
            temperature=config.TRANSLATOR_TEMPERATURE
        )
        self.parser = PydanticOutputParser(pydantic_object=TranslatedContent)

    def translate_topics(self, evaluated_topics: List[EvaluatedTopic], target_language="Korean"):
        """
        Translate evaluated topics to the target language.
        """
        print(f"Translating topics to {target_language}...")
        translated_topics = []
        
        prompt_template = """
        You are a professional academic translator. Translate the following research topic details into {target_language}.
        Ensure the tone is academic and professional.
        
        Original Title: {title}
        Original Background: {background}
        Original Necessity: {necessity}
        Original Table of Contents: {toc}
        Original Expected Effects: {expected_effects}
        Original Evaluation Reasoning: {reasoning}
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["target_language", "title", "background", "necessity", "toc", "expected_effects", "reasoning"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.parser
        
        for item in evaluated_topics:
            topic = item.topic
            eval_result = item.evaluation
            
            try:
                print(f"Translating: {topic.title}")
                translation = chain.invoke({
                    "target_language": target_language,
                    "title": topic.title,
                    "background": topic.background,
                    "necessity": topic.necessity,
                    "toc": str(topic.table_of_contents),
                    "expected_effects": topic.expected_effects,
                    "reasoning": eval_result.reasoning
                })
                
                # Create new objects with translated content
                new_topic = ResearchTopic(
                    title=translation.title,
                    background=translation.background,
                    necessity=translation.necessity,
                    table_of_contents=translation.table_of_contents,
                    expected_effects=translation.expected_effects,
                    related_papers=topic.related_papers  # Preserve the original related papers
                )
                
                new_eval = EvaluationResult(
                    originality_score=eval_result.originality_score,
                    feasibility_score=eval_result.feasibility_score,
                    impact_score=eval_result.impact_score,
                    total_score=eval_result.total_score,
                    reasoning=translation.reasoning
                )
                
                translated_topics.append(EvaluatedTopic(topic=new_topic, evaluation=new_eval))
                
            except Exception as e:
                print(f"Error translating topic '{topic.title}': {e}")
                # Fallback: keep original if translation fails
                translated_topics.append(item)
        
        print("Translation complete.")
        return translated_topics
