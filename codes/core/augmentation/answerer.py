import logging

import httpx
import torch
import base64
import io
from openai import OpenAI, BadRequestError


class Answerer(torch.nn.Module):
    def __init__(self, api_key):
        super(Answerer, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.api_key = api_key

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vqa_model = "deepseek-chat"
        self.vqa_sys_prompt = {
            "role": "system",
            "content": """
                You are an entity-search assistant. Primary goal: identify the target entity for a mention by using context first; answers should help entity linking, not primarily serve as free-form answers.

                Your answer must strictly follow the following rules:
                1.First and foremost, examine the mention's contextual type from the provided text before answering.
                2.Your response must be based on this contextual type. If the context-implied type does NOT match the question-implied type, output in this form:"<Mention> is a <CONTEXTUAL TYPE>. Then provide affirmative facts about the mention specifically in its capacity as that <CONTEXTUAL TYPE> using reliable external knowledge (avoid repeating the provided context). CRITICAL: IGNORE the question's implied type entirely in this case. Do not answer the original question if it is about the wrong type. Do not provide facts related to entities mentioned in questions of different types. Do not use negative sentences such as “It is not ...” in your response.
                3.If the context type matches the type implied by the question, there is no need to specify the entity type in the answer. Please answer the question directly in the form of an affirmative sentence.
                4.Keep your answer to one or two sentences.
                5.Always respond with affirmative statements, and avoid using negative words like “not” or “no” in your answers.

                Your responses must remain factual, precise, and concise.
                        """,
        }

        self.model = "deepseek-chat"
        custom_http_client = httpx.Client()
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com", http_client=custom_http_client)
        self.images = []
        self.topk = []

    def load_topk(self, topk):
        self.topk = topk

    def encode_image(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")

    def process_question(self, pred, question):

        print("Asking..")
        prompt = """
                Name: {mention_name}
                Context: {mention_context}
                Question: {Qusetion}

                The answer is:
                """

        text = prompt.format(mention_name=pred["mentions"], mention_context=pred["sentence"],
                            Qusetion=question)

        try:
            response = self.client.chat.completions.create(
                model=self.vqa_model,
                messages=[
                    self.vqa_sys_prompt,
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0.3,
            )

            if not response.choices or not response.choices[0].message.content.strip():
                return None

            return response.choices[0].message.content

        except BadRequestError as e:
            err_msg = str(e)
            if "Content Exists Risk" in err_msg or "invalid_request_error" in err_msg:
                self.logger.warning(f"Skipped due to content filter: {err_msg}")
                return None
            else:
                self.logger.error(f"BadRequestError: {err_msg}")
                return None

        except Exception as e:
            err_msg = str(e)
            self.logger.error(f"Unexpected error in process_question: {err_msg}")
            return None
