from typing import List, Dict, Any, Optional
import time
import os
import json
from io import BytesIO

from google import genai
from google.genai.types import JobState
from google.genai import types

from src.models.generator.base_generator import BaseGenerator
from src.registry import GENERATOR_REG


@GENERATOR_REG.register("gemini")
class GeminiGenerator(BaseGenerator):
    def __init__(self,
                 model: str = "gemini-2.5-flash"
                 ):

        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

        self.completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED
        }

        self.structure_array_dict = {

                        "response_mime_type": "application/json",
                        "response_schema": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    }

    def create_request(self, prompt: str, generation_config: Dict[str, Any]) -> Dict:

        request = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        if generation_config:
            request["config"] = generation_config

        return request

    def create_batch_request(self, prompt: str, generation_config: Dict[str, Any], system_instruction=None) -> Dict:

        request = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        if system_instruction:
            request["system_instruction"] = {"parts": [{"text": system_instruction}]}

        if generation_config:
            request["generation_config"] = generation_config

        return request

    def create_batch_job(self,
                         prompts: List[str],
                         system_instruction: str = None,
                         temperature: float = 0,
                         display_name: str = None,
                         structured_output: dict = None,
                         ):

        generation_config = {
            "temperature": temperature,
        }

        if structured_output:
            generation_config.update(structured_output)

        tasks = []
        for i, prompt in enumerate(prompts):
            task = {
                "request": self.create_batch_request(prompt, generation_config, system_instruction),
                "custom_id": str(i)
            }
            tasks.append(task)

        # Write to BytesIO (in-memory file)
        file_content = "\n".join([json.dumps(task) for task in tasks])
        file_obj = BytesIO(file_content.encode('utf-8'))
        file_obj.name = "batch_input.jsonl"

        # Upload the file
        uploaded_file = self.client.files.upload(
            file=file_obj,
            config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
        )
        print(f"Uploaded file: {uploaded_file.name}")

        # Create batch config
        batch_config = {}
        if display_name:
            batch_config['display_name'] = display_name

        # Create batch job with uploaded file
        batch_job = self.client.batches.create(
            model=self.model,
            src=uploaded_file.name,
            config=batch_config
        )


        # inline_requests = [
        #     self.create_request(prompt, generation_config)
        #     for i, prompt in enumerate(prompts)
        # ]

        # # Create batch config
        # batch_config = {}
        # if display_name:
        #     batch_config['display_name'] = display_name
        #
        # inline_batch_job = self.client.batches.create(
        #     model=self.model,
        #     src=inline_requests,
        #     config=batch_config
        # )

        return batch_job


    def wait_for_completion(self, batch_job, poll_interval: int = 10):

        while True:

            batch_job = self.client.batches.get(name=batch_job.name)

            if batch_job.state.name in  (
                'JOB_STATE_SUCCEEDED',
                'JOB_STATE_FAILED',
                'JOB_STATE_CANCELLED',
                'JOB_STATE_EXPIRED'
            ):
                break


            print(f"Job state: {batch_job.state.name}. Waiting {poll_interval}s...")
            time.sleep(poll_interval)

        print(f"Job finished with state: {batch_job.state.name}")

        return batch_job

    def get_response_batch(self, batch_job, input_length):

        results_dict = {}

        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

            # If batch job was created with a file
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                print(f"Results are in file: {result_file_name}")

                print("Downloading result file content...")
                file_content = self.client.files.download(file=result_file_name)
                # Process file_content (bytes) as needed
                content_str = file_content.decode('utf-8')
                # outputs = content_str.splitlines()

                for line in content_str.strip().split('\n'):
                    if line:
                        result_item = json.loads(line)
                        custom_id = result_item.get('custom_id')

                        # Extract response text
                        text_content = result_item['response']['candidates'][0]['content']['parts'][0]['text']
                        results_dict[int(custom_id)] = text_content


        ordered_results = [results_dict.get(i) for i in range(input_length)]
        return ordered_results

    def get_response_no_batch(self,
                              prompts: List[str],
                              system_instruction: str,
                              temperature: float = 0,
                              structured_output: dict = None,
                              ):

        responses = []

        # Build generation config
        generation_config = {
            "temperature": temperature,
        }
        if structured_output:
            generation_config.update(structured_output)

        for prompt in prompts:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        **generation_config
                    }
                )
                responses.append(response.text)
            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append(None)

        return responses


    def generate(self,
                 prompts: List[str],
                 system_instruction: str,
                 temperature:float = 0,
                 display_name: str = None,
                 in_batch=True,
                 structured_output: Optional[str] = None,
                 ) -> Dict[str, Any]:


        if structured_output == 'array':
            structured_output_dict  = self.structure_array_dict
        else:
            structured_output_dict = None


        result = {
            "status": None,
            "responses": [],
        }

        if in_batch:

            batch_job = self.create_batch_job(
                prompts=prompts,
                system_instruction=system_instruction,
                temperature=temperature,
                display_name=display_name,
                structured_output=structured_output_dict
            )

            batch_job = self.wait_for_completion(batch_job, poll_interval=5)

            responses = self.get_response_batch(batch_job, input_length=len(prompts))

            if len(responses) == 0:
                responses = [None] * len(prompts)
            result["responses"] = responses
            result["status"] = batch_job.state.name

        else:

            responses = self.get_response_no_batch(
                prompts=prompts,
                system_instruction=system_instruction,
                temperature=temperature,
                structured_output=structured_output_dict)

            result["responses"] = responses
            result["status"] = "JOB_STATE_SUCCEEDED"

        return result


if __name__ == "__main__":

    gemini = GeminiGenerator()

    p_s = ["I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. Training workers is not their job - they're building software. Perhaps educational systems in the U.S. (or their students) should worry a little about getting marketable skills in exchange for their massive investment in education, rather than getting out with thousands in student debt and then complaining that they aren't qualified to do anything."]
    ins = """Decompose the "Content" into clear statements, ensuring they are interpretable out of context.
        1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
        2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct statement.  
        3. Decontextualize the statement by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
    """

    result = gemini.generate(prompts=p_s, system_instruction=ins, structured_output='array', in_batch=False)

    print(result)