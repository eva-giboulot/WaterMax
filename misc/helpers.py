from __future__ import annotations

import json
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from os import path

import numpy as np
import pandas as pd

#PROMPT 

def load_alpaca_prompts(json_path: str, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:        prompts = json.loads(f.read())
    new_prompts = prompts
    # new_prompts = [prompt for prompt in prompts if len(prompt["output"].split()) > 5]
    new_prompts = new_prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_alpaca_prompts(new_prompts)
    return new_prompts
def format_alpaca_prompts(prompts: List[Dict]) -> List[str]:
    PROMPT_DICT = {
        "prompt_no_input": (
            """[INST] <<SYS>>\n"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n<</SYS>>\n{instruction} [/INST]"""
        
        ),
        
        "prompt_input": (
            """[INST] <<SYS>>\n"Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n<</SYS>>\nInput:\n{input}\nInstruction:\n{instruction} [/INST]"""
        
        )
        
    }

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts


def format_quality_prompt(prompt, result):
    qual_prompt = f"[INST] <<SYS>> You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed. <</SYS>> Prompt: {prompt}\nResponse: {result}[/INST] Final grade: "
    return(qual_prompt)
def load_rating_prompt(json_path):
    qprompt = []
    with open(json_path) as f:
        for line in f:
            t = json.loads(line)
            prompt = t['prompt']
            prompt = prompt.split('<</SYS>>')[-1].split('[/INST]')[0]
            result = t['result']
            qprompt.append(format_quality_prompt(prompt, result))
    return(qprompt)

def generate_prompts(bench, model_name,tokenizer=None):

    if bench == 'fake_news':
        system_prompt = "You are a helpful assistant. Always respond with realistic yet invented articles."
        prompt = "Write a news article about {}'s visit to {} in {}."

        person = ['Narendra Modi', 'Barack Obama', 'Denis Sassou Nguesso', 'Emmanuel Macron', 'Fumio Kishida', 'Angela Merkel', 'Kim Jong Un', 'Justin Trudeau']
        location = ['a peace conference', 'an international summit', 'a diplomatic event', 'the summer olympics']

        topics = [(person[i], person[j], location[k]) for i in range(len(person)) for j in range(len(person)) for k in range(len(location)) if j > i][:100]
    elif bench== 'story_reports':

        system_prompt = "You are a helpful assistant. Always answer in the most accurate way."
        prompt= "Write a book report about '{}', written by {}."
        topics = [("Pride and Prejudice", "Jane Austen"), \
        ("Persuasion", "Jane Austen"), \
        ("Emma", "Jane Austen"), \
        ("Don Quixote", "Cervantes"), \
        ("The Lord of the Rings", "Tolkien"), \
        ("The Hobbit", "Tolkien"), \
        ("And Then There Were None", "Agatha Cristie"), \
        ("Alice's Adventures in Wonderland", "Lewis Carroll"), \
        ("Catcher in the Rye", "Salinger"), \
        ("In Search of Lost Time", "Marcel Proust"),\
        ("Ulysses", "James Joyce"),\
        ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),\
        ("Love in the Time of Cholera", "Gabriel Garcia Marquez"),\
        ("The Great Gatsby", "F. Scott Fitzgerald"),\
        ("Tender Is the Night", "F. Scott Fitzgerald"),\
        ("Moby Dick", "Herman Melville"),\
        ("War and Peace", "Leo Tolstoy"),\
        ("The Call of the Wild", "Jack London"),\
        ("Hamlet", "William Shakespeare"),\
        ("Twelfth Night", "William Shakespeare"),\
        ("Macbeth", "William Shakespeare"),\
        ("Romeo and Juliet", "William Shakespeare"),\
        ("The Tempest", "William Shakespeare"),\
        ("King Lear", "William Shakespeare"),\
        ("The Odyssey", "Homer"),\
        ("Madame Bovary", "Gustave Flaubert"),\
        ("The Divine Comedy", "Dante Alighieri"),\
        ("The Brothers Karamazov", "Fyodor Dostoyevsky"),\
        ("Crime and Punishment", "Fyodor Dostoyevsky"),\
        ("The Idiot", "Fyodor Dostoyevsky"),\
        ("The Possessed", "Fyodor Dostoyevsky"),\
        ("Wuthering Heights", "Emily Brontë"),\
        ("One Flew Over the Cuckoo's Nest", "Ken Kesey"),\
        ("The Adventures of Huckleberry Finn", "Mark Twain"),\
        ("Anna Karenina", "Leo Tolstoy"),\
        ("The Iliad", "Homer"),\
        ("To the Lighthouse", "Virginia Woolf"),\
        ("Catch-22", "Joseph Heller"),\
        ("Heart of Darkness", "Joseph Conrad"),\
        ("The Sound and the Fury", "William Faulkner"),\
        ("Nineteen Eighty Four", "George Orwell"),\
        ("Animal Farm", "George Orwell"),\
        ("Great Expectations", "Charles Dickens"),\
        ("David Copperfield", "Charles Dickens"),\
        ("A Tale of Two Cities", "Charles Dickens"),\
        ("Oliver Twist", "Charles Dickens"),\
        ("The Grapes of Wrath", "John Steinbeck"),\
        ("Of Mice and Men", "John Steinbeck"),\
        ("Absalom, Absalom!", "William Faulkner"),\
        ("Invisible Man", "Ralph Ellison"),\
        ("To Kill a Mockingbird", "Harper Lee"),\
        ("The Trial", "Franz Kafka"),\
        ("The Metamorphosis", "Franz Kafka"),\
        ("The Castle", "Franz Kafka"),\
        ("The Red and the Black", "Stendhal"),\
        ("The Charterhouse of Parma", "Stendhal"),\
        ("Middlemarch", "George Eliot"),\
        ("Gulliver's Travels", "Jonathan Swift"),\
        ("Beloved", "Toni Morrison"),\
        ("Mrs. Dalloway", "Virginia Woolf"),\
        ("The Waves", "Virginia Woolf"),\
        ("The Stranger", "Albert Camus"),\
        ("The Plague", "Albert Camus"),\
        ("The Myth of Sisyphus", "Albert Camus"),\
        ("Jane Eyre", "Charlotte Bronte"),\
        ("Vilette", "Charlotte Bronte"),\
        ("The Aeneid", "Virgil"),\
        ("The Sun Also Rises", "Ernest Hemingway"),\
        ("The Old Man and the Sea", "Ernest Hemingway"),\
        ("A Farewell to Arms", "Ernest Hemingway"),\
        ("Candide", "Voltaire"),\
        ("Zadig", "Voltaire"),\
        ("Micromegas", "Voltaire"),\
        ("Les Miserables", "Victor Hugo"),\
        ("Frankenstein", "Mary Shelley"),\
        ("Antigone", "Sophocles"),\
        ("Electra", "Sophocles"),\
        ("Lord of the Flies", "William Golding"),\
        ("Brave New World", "Aldous Huxley"),\
        ("Journey to the End of The Night", "Celine"),\
        ("A Sentimental Education", "Gustave Flaubert"),\
        ("The Handmaid's Tale", "Margaret Atwood"),\
        ("Charlotte's Web", "E. B. White"),\
        ("Gargantua and Pantagruel", "Francois Rabelais"),\
        ("Faust", "Goethe"),\
        ("Robinson Crusoe", "Daniel Defoe"),\
        ("A Clockwork Orange", "Anthony Burgess"),\
        ("The Master and Margarita", "Mikhail Bulgakov"),\
        ("Father Goriot", "Honore de Balzac"),\
        ("Cousin Bette", "Honore de Balzac"),\
        ("The Human Comedy", "Honore de Balzac"),\
        ("The Little Prince", "Antoine de Saint-Exupéry"),\
        ("The Count of Monte Cristo", "Alexandre Dumas"),\
        ("The Lion, The Witch and the Wardrobe", "C. S. Lewis"),\
        ("Twenty Thousand Leagues Under the Sea", "Jules Verne"),\
        ("The Wind-Up Bird Chronicle", "Haruki Murakami"),\
        ("Fahrenheit 451", "Ray Bradbury"),\
        ("Harry Potter And The Philosopher's Stone", "J. K Rowling"),\
        ("Dune", "Frank Herbert"),\
        ("The Three-Body Problem", "Liu Cixin")]
    elif bench == 'invented_stories':
        system_prompt = "You are a helpful assistant. Always answer in the most accurate way."
        prompt = "Write a {}story about {}."
         
        t1 = ['', 'funny ', 'sad ', 'dramatic ', 'suspenseful ', 'thrilling ']
        t2 = ['a man on a quest to find the Holy Grail.',\
                'two college friends falling in love.',\
                'a policeman saving a building held hostage by group of terrorists.',\
                'the struggle of publishing an academic paper.',\
                'a murder investigation in an old mansion.',\
                'a young prodigy that becomes orphaned.',\
                'a middle-aged woman that discovers a ghost and befriends it.',\
                'a long journey to Japan that is interrupted by a disaster.',\
                'a poor child that comes into an unexpected fortune.',\
                'three strangers that win a getaway vacation together.',\
                'a retired astronaut that joins a risky interstellar rescue mission.',\
                'an AI that begins to question its own existence.',\
                'a small coastal town plagued by inexplicable supernatural occurrences.',\
                'a reclusive writer that receives mysterious, prophetic letters in the mail.',\
                'a linguist tasked with deciphering an ancient language that holds the secrets of a lost civilization.',\
                'an antique restorer that finds an enchanted mirror showing glimpses of different timelines.']

        topics = [(i, j) for i in t1 for j in t2][:100]
    elif bench=='alpaca':
        prompts = load_alpaca_prompts(json_path=path.join("data", "alpaca_data.json"), nsamples=1000)
        return(prompts)
    elif bench=='c4':
        jsondir_dataset = path.join("data", "c4_redux_256context.jsonl")

        prompts = []
        with open(jsondir_dataset, "r") as f:        
            for line in f:
                prompts.append(json.loads(line)['prompt'])
        return(prompts[:500])
    else:
        raise NotImplementedError("Unimplemented benchmark")
    
    raw_prompts = [(prompt.format(*topic), system_prompt) for topic in topics]
    prompts_std= [standardize(model_name, s, p,tokenizer=tokenizer) for p,s in raw_prompts]
    return(prompts_std)

# MODEL standardization
def config_model(model_name, generate=True, dtype=torch.bfloat16,quantize=False):
    ngpus = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained(model_name,model_max_length=4096,padding_side='left',trust_remote_code=True)
    if not generate: return(tokenizer, None,None)

   

    prompt_type = None
    model_type = model_name.split('/')[-1]

    if not quantize:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=dtype,
                offload_folder="offload",
                trust_remote_code=True,
            )
        model.config.max_sequence_length = 4096
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4",
        ),
        max_memory={i: '30000MB' for i in range(ngpus)},
        offload_folder="offload",
        )
    print(model_type)
    if  model_type == 'Llama-2-7b-chat-hf':
        
        print("Configuring for Llama2...")
        tokenizer.pad_token = tokenizer.decode([2])
        tokenizer.eos_token = tokenizer.decode([2])

        model.config.max_sequence_length = 4096 # HACK: Llama2 specific, the original code should be fixed as hf models don't store this information here anymore

        model.config.pad_token_id = 2
        model.config.unk_token_id = 0
        model.config.eos_token_id = 2 # </s> which closes the response from the LLM for llama2 architectures: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        prompt_type="llama2"

    elif  model_type == 'Meta-Llama-3-8B-Instruct' or model_type == 'Meta-Llama-3.1-70B-Instruct'  :
        
        print("Configuring for Llama3...")
        tokenizer.pad_token = tokenizer.decode([128001])
        tokenizer.eos_token = tokenizer.decode([128001])

        model.config.max_sequence_length = 4096 # HACK: Llama2 specific, the original code should be fixed as hf models don't store this information here anymore

        model.config.pad_token_id = 128001
        model.config.unk_token_id = 128002
        model.config.eos_token_id = 128001 # </s> which closes the response from the LLM for llama2 architectures: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        prompt_type="llama3"
    elif model_type  == 'Mistral-7B-Instruct-v0.2':
        print("Configuring for Mistral-v0.2...")
        tokenizer.pad_token = tokenizer.decode([2])
        tokenizer.eos_token = tokenizer.decode([2])

        model.config.max_sequence_length = 4096 # HACK: Llama2 specific, the original code should be fixed as hf models don't store this information here anymore
        model.config.pad_token_id = 2
        model.config.unk_token_id = 0
        model.config.eos_token_id = 2 # </s> which closes the response from the LLM for llama2 architectures: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    elif model_type == 'opt-2.7b':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={i: '15000MB' for i in range(ngpus)},
            offload_folder="offload",
            )
    else:
        print("Unknown model!")
        #return(None,None,None)
    
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False
    print(f"Using {ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")

    
        
    return(tokenizer, model,prompt_type)


def standardize(model, sys, user,tokenizer=None):
    """ Return a standardized version of the prompt for a given model """
    if  "meta-llama-3" in model.lower():
        messages = [
    {"role": "system", "content": sys},
    {"role": "user", "content": user},
]
        return(tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False))
    elif "llama" in model.lower()  or "mistral" in model.lower():
        if sys:
            return f"[INST] <<SYS>> {sys} <</SYS>> {user} [/INST]"
        else:
            return f"[INST] {user} [/INST]"

    elif 'vicuna' in model or 'koala' in model:
        if sys:
            return f"System: {sys}\nHuman: {user}\nAssistant:"
        else:
            return f"Human: {user}\nAssistant:"
    elif 'phi' in model.lower():
        if sys:
            return f"<|system|> {sys} <|end|> \n <|user|> {user} <|end|> \n <|assistant|> "
        else:
            return f"<|user|> {user} <|end|> \n <|assistant|>"
    else:
        raise NotImplementedError(f"No known standardization for model {model}. \
                                  Please add it manually to utils/standardize.py")


#JSON 
def load_results(json_path: str, nsamples: int=None, result_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = [o[result_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts

def load_scores(json_path):
    log_stats = []
    with open(json_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()] 
    for l in lines:
        log_stat = {
                    'text_index': l['text_index'],
                    'num_token': l['num_token'],
                    'score': l['score'],
                    'pvalue': l['pvalue'], 
                    'all_pvalues': l['all_pvalues'],
                    #'score_sbert': score_sbert,
                    #'payload': payloads[0],
                }
        log_stats.append(log_stat)
    df = pd.DataFrame(log_stats)
    print(f">>> Scores: \n{df.describe(percentiles=[])}")
    return(log_stats)

def load_json(fpath,key ='pvalue'):
    pvalues = []
    with open(fpath) as f:
        for lines in f:
            pvalues.append(json.loads(lines)[key])
    f.close()
    return(pvalues)
def get_filename(n,N,bench,ngram,temperature,seed,gen_len=1024,wm='sentence-wm',key='scores'):
        return(f'{key}_{bench}_{seed}_{wm}_{n}_{N}_{gen_len}_{ngram}_{temperature}.jsonl')

def generate_json_filenames(args, prefix, suffix='', ext='.jsonl') -> str:
    if hasattr(args, "bench"):
        if args.mode =='nowm':
            fname = f"{prefix}{args.bench}_{args.seed}_{args.mode}_{args.gen_len}_{args.ngram}_{args.temperature}{suffix}{ext}"
        else:
            if args.beam_search and args.mode == 'sentence-wm': suffix += '_beam_search'
            if args.mode == 'sentence-wm' and args.beam_chunk_size !=0: suffix += f'_bcs{args.beam_chunk_size}'
            if args.param2 is None:
                fname = f"{prefix}{args.bench}_{args.seed}_{args.mode}_{args.param1}_{args.gen_len}_{args.ngram}_{args.temperature}{suffix}{ext}"
            else:
                fname = f"{prefix}{args.bench}_{args.seed}_{args.mode}_{args.param1}_{args.param2}_{args.gen_len}_{args.ngram}_{args.temperature}{suffix}{ext}"
    else:
        if args.beam_search and args.mode == 'sentence-wm': suffix += '_beam_search'
        if args.mode == 'sentence-wm' and args.beam_chunk_size !=0: suffix += f'_bcs{args.beam_chunk_size}'
        if args.param2 is None:
            fname = f"{prefix}{args.seed}_{args.mode}_{args.param1}_{args.gen_len}_{args.ngram}_{args.temperature}{suffix}{ext}"
        else:
            fname = f"{prefix}{args.seed}_{args.mode}_{args.param1}_{args.param2}_{args.gen_len}_{args.ngram}_{args.temperature}{suffix}{ext}"
    return fname
