from typing import List, Dict, Any, Optional
import time
import os
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai.types import JobState
from google.genai import types

from src.models.generator.base_generator import BaseGenerator
from src.registry import GENERATOR_REG


@GENERATOR_REG.register("gemini")
class GeminiGenerator(BaseGenerator):
    def __init__(self,
                 model: str = "gemini-2.5-flash-lite"
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
                         top_k: int = None,
                         top_p: float = None,
                         display_name: str = None,
                         structured_output: dict = None,
                         seed: int = 42,
                         ):

        generation_config = {
            "temperature": temperature,
            "seed": seed
        }
        if top_k is not None:
            generation_config["top_k"] = top_k
        if top_p is not None:
            generation_config["top_p"] = top_p

        if structured_output:
            generation_config.update(structured_output)

        print(f"Generation configuration: {generation_config}")

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
            config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='application/json')
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

                        try:
                            # Extract response text
                            text_content = result_item['response']['candidates'][0]['content']['parts'][0]['text']

                        except KeyError as e:
                            print("text_content", e)
                            text_content = ""

                        results_dict[int(custom_id)] = text_content


        ordered_results = [results_dict.get(i) for i in range(input_length)]
        return ordered_results

    def _call_model(self,
                    prompt: str,
                    system_instruction: str,
                    generation_config: Dict[str, Any]) -> Optional[str]:

        if system_instruction:
            config = {
                "system_instruction": system_instruction,
                **generation_config
            }
        else:
            config = generation_config

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text

    def get_response_no_batch(self,
                              prompts: List[str],
                              system_instruction: str,
                              temperature: float = 0,
                              top_k: int = None,
                              top_p: float = None,
                              structured_output: dict = None,
                              max_workers: Optional[int] = None,
                              seed: int = 42,
                              ):

        if not prompts:
            return []

        # Build generation config
        generation_config = {
            "temperature": temperature,
            "seed": seed
        }
        if top_k is not None:
            generation_config["top_k"] = top_k
        if top_p is not None:
            generation_config["top_p"] = top_p

        # print(f"Generation configuration: {generation_config}, System instruction: {system_instruction}")

        if structured_output:
            generation_config.update(structured_output)

        max_workers = max_workers or 1
        responses: List[Optional[str]] = [None] * len(prompts)

        for idx, prompt in enumerate(prompts):
            try:
                responses[idx] = self._call_model(prompt, system_instruction, generation_config)
            except Exception as e:
                print(f"Error generating response: {e}")
                responses[idx] = None
        return responses


        # if max_workers <= 1:
        #     for idx, prompt in enumerate(prompts):
        #         try:
        #             responses[idx] = self._call_model(prompt, system_instruction, generation_config)
        #         except Exception as e:
        #             print(f"Error generating response: {e}")
        #             responses[idx] = None
        #     return responses
        #
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     future_to_idx = {
        #         executor.submit(self._call_model, prompt, system_instruction, generation_config): idx
        #         for idx, prompt in enumerate(prompts)
        #     }
        #
        #     for future in as_completed(future_to_idx):
        #         idx = future_to_idx[future]
        #         try:
        #             responses[idx] = future.result()
        #         except Exception as e:
        #             print(f"Error generating response: {e}")
        #             responses[idx] = None
        #
        # return responses


    def generate(self,
                 prompts: List[str],
                 system_instruction: str=None,
                 temperature:float = 0,
                 top_k:int = None,
                 top_p:float = None,
                 display_name: str = None,
                 in_batch=True,
                 structured_output: Optional[str] = None,
                 max_workers: Optional[int] = None,
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
                top_k=top_k,
                top_p=top_p,
                display_name=display_name,
                structured_output=structured_output_dict,
                seed=42
            )

            batch_job = self.wait_for_completion(batch_job, poll_interval=30)

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
                top_k=top_k,
                top_p=top_p,
                structured_output=structured_output_dict,
                max_workers=max_workers,
                seed=42
            )

            result["responses"] = responses
            result["status"] = "JOB_STATE_SUCCEEDED"

        return result


if __name__ == "__main__":

    gemini = GeminiGenerator(model="gemini-2.5-flash")

    p_s = ["I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. "
           "Training workers is not their job - they're building software. Perhaps educational systems in the U.S. "] * 10
    ins = """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.

Input: Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America." ]"""

    new_prompts = [ins + "\n\n" + f"Input: {p}" + "\nOutput:" for p in p_s]

    # result = gemini.generate(
    #     prompts=new_prompts,
    #     structured_output='array',
    #     in_batch=False,
    #     temperature=0,
    #     top_k=1,
    #     max_workers = 4
    # )
    #
    # print(result)

    print(new_prompts[0])
