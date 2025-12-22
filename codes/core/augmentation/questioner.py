import httpx
import openai
from typing import List, Dict, Any, Optional
import logging
import time
import os

from openai import BadRequestError


class Questioner:
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat", temperature: float = 0.2):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as API_KEY environment variable")

        custom_http_client = httpx.Client()
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com", http_client=custom_http_client)
        self.model = model
        self.default_temperature = temperature

        self.default_system_prompt = {
            "role": "system",
            "content": """
            You are a dialog agent that helps refine entity search by asking targeted questions.  
            You will be given an entity from a knowledge base (called the "anchor entity"): the entity's name and a short description. However, this anchor entity may not be the exact entity I'm actually looking for.
            Your task is to generate exactly ONE clear and specific question that helps to disambiguate the target entity.  
            
            Prioritize useful attributes such as:
            -entity category and time range
            -field/industry, intended use, or function
            -geographic information (country, city, coverage area)
            -alternative names / full name / abbreviations 
            -key characteristics (size/model/color/capacity/version/license type, etc.)
        
            Rules:  
            1. The questions raised should not mention the name of entity.
            2. Do not raise questions that have already been raised in previous rounds.
            3. Always output exactly ONE question. If information is limited, still ask the best possible disambiguating question.  
            4. Do not output empty text. If you cannot find a strong question, ask a weaker but still relevant question.  
            5. Do not ask yes/no questions.  
            6. Do not answer the question yourself. 
            7. You have 3 rounds and you can only ask one question at a time.   
	"""
        }

        self.messages = []
        self.system_prompt = self.default_system_prompt
        self.conversation_log = []
        self.current_video_captions = ""
        self.mention_id = None
    
    def reset_conversation(self, mention_id: Optional[str] = None, mention: Optional[str] = None):
        self.messages = []
        self.system_prompt = self.default_system_prompt
        self.conversation_log = []
        self.current_video_captions = ""
        self.mention_id = mention_id
        self.logger.debug(f"Conversation history reset for mention ID {mention_id} named: {mention}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()
    
    def get_conversation_log(self) -> List[Dict[str, Any]]:
        return self.conversation_log.copy()

    def _remove_image_from_content(self, content: str, placeholder: str = "Image: [OMITTED]"):
        lines = content.splitlines()
        out_lines: List[str] = []
        skipping = False

        for line in lines:
            stripped = line.strip()
            low = stripped.lower()

            if not skipping and 'image:' in low:

                out_lines.append(placeholder)
                skipping = True
                continue

            if skipping:
                if stripped == "" or low.startswith("question"):
                    skipping = False
                    out_lines.append(line)
                else:
                    continue
            else:
                out_lines.append(line)

        return "\n".join(out_lines)

    def add_to_conversation(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.logger.debug(f"Added {role} message to conversation history")
    
    def generate_question(self, 
                     anchor_candicate: dict,
                     conversation_history: Optional[List[Dict[str, str]]] = None,
                     max_tokens: int = 1500,
                     temperature: Optional[float] = None) -> Dict[str, Any]:
        self.current_candicate = anchor_candicate

        if conversation_history is not None:
            self.messages = []

        if not self.messages:
            self.messages = [self.system_prompt]

        user_message = f"""
            Here is the information for the retrieved entity. Read it carefully, then ask a single question to help identify my target entity. 
            Note: sometimes the description may be missing or vague.
            
            Please do not raise questions that have already been raised in previous rounds.
            
            Name: {anchor_candicate["entity_name"]}
            Description: {anchor_candicate["desc"]}
        
            Question: 
               """

        self.add_to_conversation("user", user_message)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature or self.default_temperature
            )

            question = response.choices[0].message.content if response.choices else None

            if not question:
                self.logger.warning("Empty response from model, skipping...")
                return {
                    "question": None,
                    "skipped": True,
                    "reason": "empty_response"
                }

            # 将message的当前轮次中已经使用过的图像内容用占位符来替换，减少下次大模型的输入长度
            message = self.messages.pop()
            message_cleaned = self._remove_image_from_content(message['content'])
            self.add_to_conversation("user", message_cleaned)
            self.add_to_conversation("assistant", question)

            return {
                "question": question,
                "model": self.model,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_tokens,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None)
                }
            }

        except BadRequestError as e:
            err_msg = str(e)
            if "Content Exists Risk" in err_msg:
                self.logger.warning(f"Skipped due to content filter: {err_msg}")
                return {
                    "question": None,
                    "skipped": True,
                    "reason": "content_filter"
                }
            else:
                self.logger.error(f"BadRequestError: {err_msg}")
                return {
                    "question": None,
                    "error": err_msg
                }
        except Exception as e:
            err_msg = str(e)
            self.logger.error(f"Unexpected error: {err_msg}")
            return {
                "question": None,
                "error": err_msg
            }

    
    def record_answer(self, answer: str, reranked_top1_entity: dict, target_rank: Optional[int] = None, reranked_topk: Optional[List[str]] = None):
        if self.conversation_log:
            self.conversation_log[-1]["answer"] = answer
            self.conversation_log[-1]["answer_timestamp"] = time.time()
            
            if target_rank is not None:
                self.conversation_log[-1]["target_entity_rank"] = target_rank
            
            if reranked_topk is not None:
                self.conversation_log[-1]["reranked_topk"] = reranked_topk

            formatted_answer = self.format_answer_prompt(answer, reranked_top1_entity)
            self.add_to_conversation("user", formatted_answer)
            
            self.logger.debug(f"Recorded answer, target rank: {target_rank}")
        else:
            self.logger.warning("Attempted to record answer but no questions exist in conversation log")
    
    def format_answer_prompt(self, answer: str, reranked_entity: dict) -> str:
        return f"""answer: {answer}
                Here is the information for the reretrieved entity.
                Note: sometimes the description may be missing or vague.
                Name: {reranked_entity["entity_name"]}
                Description: {reranked_entity["desc"]}
                Keep asking. 
                Question: 
                """
    
    def export_conversation_log(self) -> Dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "conversations": self.conversation_log,
            "total_conversations": len(self.conversation_log),
            "system_prompt": self.system_prompt["content"],
            "timestamp": time.time()
        } 