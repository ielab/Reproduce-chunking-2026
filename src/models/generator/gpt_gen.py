from typing import List, Optional, Dict, Any
import json
import time
import os

from openai import OpenAI
from io import BytesIO
from pydantic import BaseModel

from src.models.generator.base_generator import BaseGenerator
from src.registry import GENERATOR_REG


list_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "item_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "category": {"type": "string"},
                            "priority": {"type": "integer"}
                        },
                        "required": ["name", "category", "priority"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": False
        }
    }
}


class ResponseEntryList(BaseModel):
    entries: List[str]



@GENERATOR_REG.register("gpt")
class GPTGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4.1-nano-2025-04-14"):

        api_key = os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)

        self.model = model
        self.list_schema = list_schema

    def create_batch_job(self,
                         prompts: List[str],
                         system_instruction: str = None,
                         temperature: float = 0,
                         structured_output: dict = None
                         ):

        tasks = []

        for idx, prompt in enumerate(prompts):
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }

            if structured_output is not None:
                body["response_format"] = structured_output

            task = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            tasks.append(task)

        # Write to BytesIO (in-memory file)
        file_content = "\n".join([json.dumps(task) for task in tasks])
        file_obj = BytesIO(file_content.encode('utf-8'))
        file_obj.name = "batch_input.jsonl"

        # Upload from memory
        batch_file = self.client.files.create(
            file=file_obj,
            purpose="batch"
        )

        # Create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"Batch job created: {batch_job.id}")
        print(f"Status: {batch_job.status}")

        return batch_job

    def wait_for_completion(self, batch_job, poll_interval: int = 10):

        # 3. POLL FOR COMPLETION
        while True:

            batch_job = self.client.batches.retrieve(batch_job.id)
            print(f"Status: {batch_job.status}")

            if batch_job.status == "completed":
                print("Batch completed!")
                break

            elif batch_job.status in ["failed", "expired", "cancelled"]:
                print(f"Batch {batch_job.status}")

                # Get error details
                if hasattr(batch_job, 'errors') and batch_job.errors:
                    print(f"\nBatch-level errors: {batch_job.errors}")

                break

            print(f"Job state: {batch_job.status}. Waiting {poll_interval}s...")
            time.sleep(poll_interval)

        return batch_job

    def get_response_batch(self, batch_job, input_length: int):

        ordered_results = []

        if batch_job.status == "completed":

            result_file_id = batch_job.output_file_id
            result_content = self.client.files.content(result_file_id).content


            results_dict = {}
            for line in result_content.decode('utf-8').strip().split('\n'):
                json_obj = json.loads(line)
                custom_id = int(json_obj['custom_id'])

                if json_obj.get('error'):
                    results_dict[custom_id] = None
                else:
                    response = json_obj['response']['body']['choices'][0]['message']['content']
                    results_dict[custom_id] = response

            ordered_results = [results_dict.get(i) for i in range(input_length)]

        return ordered_results

    def get_response_no_batch(self,
                              prompts: List[str],
                              system_instruction: str = None,
                              temperature: float = 0,
                              structured_output: dict = None
                              ):

        responses = []

        for prompt in prompts:
            try:
                if structured_output is not None:
                    response = self.client.responses.parse(
                        model=self.model,
                        input=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        text_format=ResponseEntryList
                    )

                    content = response.output_parsed.entries

                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                    )

                    content = response.output_text
                responses.append(content)

            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append(None)


        return responses


    def generate(self,
                 prompts: List[str],
                 system_instruction: str,
                 temperature: float = 0,
                 display_name: str = None,
                 in_batch=True,
                 structured_output: Optional[str] = None,
                 ) -> Dict[str, Any]:

        result = {
            "status": None,
            "responses": [],
        }


        if in_batch:

            if structured_output == "array":
                structured_output_dict = self.list_schema
            else:
                structured_output_dict = None

            batch_job = self.create_batch_job(
                prompts=prompts,
                system_instruction=system_instruction,
                temperature=temperature,
                structured_output=structured_output_dict
            )

            batch_job = self.wait_for_completion(batch_job, poll_interval=10)
            responses = self.get_response_batch(batch_job, input_length=len(prompts))

            result["responses"] = responses
            result["status"] = batch_job.state.name

        else:

            responses = self.get_response_no_batch(
                prompts=prompts,
                system_instruction=system_instruction,
                temperature=temperature,
                structured_output=structured_output
            )

            result["responses"] = responses
            result["status"] = "completed"

        return result


if __name__ == "__main__":

    gen_model = GPTGenerator()

    p_s =["I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. Training workers is not their job - they're building software. Perhaps educational systems in the U.S. (or their students) should worry a little about getting marketable skills in exchange for their massive investment in education, rather than getting out with thousands in student debt and then complaining that they aren't qualified to do anything."]
    ins = """Decompose the "Content" into clear statements, ensuring they are interpretable out of context.
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct statement.  
    3. Decontextualize the statement by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
"""

    result = gen_model.generate(prompts=p_s, system_instruction=ins, structured_output='array', in_batch=False)

    print(result)






