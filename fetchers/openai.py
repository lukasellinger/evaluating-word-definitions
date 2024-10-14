"""Module for making api call to OpenAi."""
import os
from typing import List

from openai import OpenAI
from openai.types import FileObject, Batch
from openai.types.chat import ChatCompletion

from config import OPEN_AI_TOKEN


class OpenAiFetcher:
    """Wrapper for OpenAi api calls."""

    def __init__(self, api_key=None):
        self.api_key = api_key or OPEN_AI_TOKEN
        self.client = OpenAI(api_key=self.api_key)
        self.current_batch_jobs = {}

    def upload_batch_file(self, file_name: str) -> FileObject:
        """
        Uploads a file to the OpenAI API for batch processing.

        :param file_name: Path to the file that needs to be uploaded.
        :return: The uploaded batch file's metadata, including the file ID.
        """
        with open(file_name, "rb") as file:
            batch_file = self.client.files.create(
                file=file,
                purpose="batch"
            )
        return batch_file

    def create_batch_job(self, file_name: str, endpoint="/v1/chat/completions") -> Batch:
        """
        Creates a batch job using the file at file_name and a specified endpoint.

        :param file_name: Path to the file to be processed in the batch job.
        :param endpoint: The API endpoint to send the batch request to. Defaults to
                         "/v1/chat/completions".
        :return: Metadata of the created batch job, including job ID.
        """
        batch_file = self.upload_batch_file(file_name)
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=endpoint,
            completion_window="24h"
        )
        return batch_job

    def get_batch_result(self, output_file: str, batch_job: Batch) -> str | None:
        """
        Retrieves the result of a completed batch job and saves it to a specified file.

        :param output_file: Path where the result file should be saved.
        :param batch_job: Metadata of the batch job from which to retrieve the result.
        :return: Path to the output file where the result was saved, or None if no result is
        available.
        """
        result_file_id = batch_job.output_file_id
        if not result_file_id:
            return

        result = self.client.files.content(result_file_id).content

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as file:
            file.write(result)

        return output_file

    def get_batch_update(self, batch_job: Batch) -> Batch:
        """
        Retrieves the status update for a given batch job.

        :param batch_job: for which to retrieve the status update.
        :return: updated batch_job
        """
        return self.client.batches.retrieve(batch_job.id)

    def get_output(self, messages: List, model="gpt-4o-mini", temperature=0.1) -> ChatCompletion:
        """
        Generates a response from OpenAI's models based on the provided messages.

        :param messages: List of message dictionaries to be sent to the OpenAI model.
        :param model: The model to be used for generating the response. Defaults to "gpt-4o-mini".
        :param temperature: Sampling temperature for response generation. Lower values make the
                            output more deterministic. Defaults to 0.1.
        :return: The response generated by the model, including the message and log probabilities.
        """
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            seed=42,
            logprobs=True,
            top_logprobs=5
        )

        return response
