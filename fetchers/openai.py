import os

from openai import OpenAI

from config import OPEN_AI_TOKEN


class OpenAiFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or OPEN_AI_TOKEN
        self.client = OpenAI(api_key=self.api_key)
        self.current_batch_jobs = {}

    def upload_batch_file(self, file_name):
        with open(file_name, "rb") as file:
            batch_file = self.client.files.create(
                file=file,
                purpose="batch"
            )
        return batch_file

    def create_batch_job(self, file_name, endpoint="/v1/chat/completions"):
        batch_file = self.upload_batch_file(file_name)
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=endpoint,
            completion_window="24h"
        )
        return batch_job

    def get_batch_result(self, output_file, batch_job):
        result_file_id = batch_job.output_file_id
        if not result_file_id:
            return None

        result = self.client.files.content(result_file_id).content

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as file:
            file.write(result)

        return output_file

    def get_batch_update(self, batch_job):
        return self.client.batches.retrieve(batch_job.id)

    def get_output(self, messages, model="gpt-4o-mini", temperature=0.1):
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            seed=42,
            logprobs=True,
            top_logprobs=5
        )

        return response
