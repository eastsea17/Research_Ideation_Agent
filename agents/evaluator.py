from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import config
from agents.generator import ResearchTopic

class EvaluationResult(BaseModel):
    originality_score: int = Field(description="Score for originality (1-5)")
    feasibility_score: int = Field(description="Score for feasibility (1-5)")
    impact_score: int = Field(description="Score for impact (1-5)")
    total_score: int = Field(description="Sum of all scores")
    reasoning: str = Field(description="Reasoning for the scores")

class EvaluatedTopic(BaseModel):
    topic: ResearchTopic
    evaluation: EvaluationResult

class EvaluationOutput(BaseModel):
    evaluations: List[EvaluatedTopic]

class TopicEvaluator:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.MODEL_EVALUATOR,
            temperature=config.EVALUATOR_TEMPERATURE
        )
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    def evaluate_topics(self, topics: List[ResearchTopic]):
        """
        Evaluate and prioritize research topics.
        """
        print("Evaluating topics...")
        evaluated_topics = []
        
        prompt_template = """
        You are a senior research committee member. Evaluate the following research topic based on three criteria:
        1. Originality (1-5): How novel is the idea?
        2. Feasibility (1-5): Is it realistic to implement?
        3. Impact (1-5): What is the potential contribution?
        
        Topic Title: {title}
        Background: {background}
        Necessity: {necessity}
        Expected Effects: {expected_effects}
        
        Provide a score for each and a brief reasoning.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["title", "background", "necessity", "expected_effects"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.parser
        
        for topic in topics:
            try:
                print(f"Evaluating: {topic.title}")
                evaluation = chain.invoke({
                    "title": topic.title,
                    "background": topic.background,
                    "necessity": topic.necessity,
                    "expected_effects": topic.expected_effects
                })
                
                # Calculate total score if not provided correctly by LLM (though Pydantic should handle it if prompted right)
                if evaluation.total_score == 0:
                    evaluation.total_score = evaluation.originality_score + evaluation.feasibility_score + evaluation.impact_score
                
                evaluated_topics.append(EvaluatedTopic(topic=topic, evaluation=evaluation))
            except Exception as e:
                print(f"Error evaluating topic '{topic.title}': {e}")
                # Assign default low score on error
                default_eval = EvaluationResult(
                    originality_score=1, feasibility_score=1, impact_score=1, 
                    total_score=3, reasoning="Evaluation failed."
                )
                evaluated_topics.append(EvaluatedTopic(topic=topic, evaluation=default_eval))
        
        # Sort by total score descending
        evaluated_topics.sort(key=lambda x: x.evaluation.total_score, reverse=True)
        
        print("Evaluation complete.")
        return evaluated_topics
