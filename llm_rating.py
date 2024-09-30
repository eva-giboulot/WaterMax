import numpy as np
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from os import path, listdir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
    4. Please do not generate any other opening, closing, and explanations.

    ###The instruction to evaluate:
    {}

    ###Response to evaluate:
    {}

    ###Reference Answer (Score 5):
    {}

    ###Score Rubrics:
    {}

    ###Feedback: """
RUBRIC = "Grade the answer in terms of level of details, and typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed."



model = AutoModelForCausalLM.from_pretrained("/home/qgiboulot/repos/prometheus-7b-v2.0",
                                             device_map="auto",
                                            torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(   "/home/qgiboulot/repos/prometheus-7b-v2.0")




def generate_rating_prompt(instruction,reference, result ):
    user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(instruction,result, reference, RUBRIC) # Fill the prompt with your data

    messages = [
        {"role": "user", "content": user_content},
    ]

    return(messages)