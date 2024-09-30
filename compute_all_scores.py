from os import path
import argparse

from models.wm import MarylandDetector,SecureGaussianSentenceWm,GaussianSentenceWm,OpenaiDetector,NewGaussianSentenceWm
from misc.wm_dataclasses import WmConfig, int_or_float, opt_int_or_float
from misc.helpers import load_json,get_filename,config_model,generate_json_filenames


import numpy as np
from typing  import Union
import tqdm
import pickle

#python compute_all_scores.py --wm=aaron --seed=1015 --ngram=4 --model_name=[PATH_TO_MODEL]/Mistral-7B-Instruct-v0.2 --param1=5.0 --param2=0.5 --bench=fake_news

#python compute_all_scores.py --wm=kirch --seed=1015 --ngram=4 --model_name=[PATH_TO_MODEL]/Mistral-7B-Instruct-v0.2 --param1=5.0 --param2=0.5 --bench=fake_news
#python compute_all_scores.py --wm=sentence-wm --seed=1015 --ngram=4 --model_name=[PATH_TO_MODEL]/Mistral-7B-Instruct-v0.2 --param1=4 --param2=16 --bench=fake_news

def parse_arguments():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--wm', type=str, default='kirch')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ngram', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='[PATH_TO_MODEL]/Llama-2-7b-chat-hf')
    parser.add_argument('--res_path', type=str, default='./results/benchmark')


    parser.add_argument('--param1', type=int_or_float) 
    parser.add_argument('--param2', type=opt_int_or_float, default=None) 
    parser.add_argument('--beam_chunk_size', type=int, default=0) 

    parser.add_argument('--temperature', type=float, default=1.) #Temperature

    parser.add_argument('--gen_len', type=int, default=1024) #Size of the generated text
    parser.add_argument('--robust', action=argparse.BooleanOptionalAction)
    parser.add_argument('--bench', type=str, default='story_reports')





    return parser

#HELPER
def get_detector(tokenizer, config,robust=False):
    if config.wm == 'kirch':
        scoring_method = 'v2'
        detector = MarylandDetector(tokenizer, ngram=config.ngram, delta=config.param1, gamma=config.param2, seed=config.seed)
    
    elif config.wm == 'aaronson':
        scoring_method = 'v2'
        detector = OpenaiDetector(tokenizer, ngram=config.ngram, seed=config.seed)
    elif config.wm == 'robust-sentence-wm':
        scoring_method = 'v1'
   
    elif config.wm == 'sentence-wm':
        scoring_method = 'v1'
        if  robust:
            detector = SecureGaussianSentenceWm(tokenizer, ngram=config.ngram, seed=config.seed)
        else:
            detector = NewGaussianSentenceWm(tokenizer, ngram=config.ngram, seed=config.seed,split_len=config.gen_len//config.param2)
    else:
        raise Exception("Unknown watermark scheme")
    
    return(detector, scoring_method)

#MAIN
def compute_all_scores(detector, scoring_method, config,res_path,outpath):
    fpath = generate_json_filenames(config, 'results_', suffix='')
    texts = load_json(path.join(res_path, fpath),key='result')
    
    
    scores_no_aggreg = []
    for i,text in enumerate(tqdm.tqdm(texts)):#tqdm.tqdm(zip(results, results_orig)):
            # compute watermark score
            scores = np.array(detector.get_scores_by_t([text], scoring_method=scoring_method)).ravel()
            #print(scores)
            scores_no_aggreg.append(scores)
    return(scores_no_aggreg)


def main():
    args = parse_arguments().parse_args()
    res_path = path.join(args.res_path,args.model_name.split('/')[-1]) 
    config = WmConfig(seed=args.seed, param1=args.param1, param2=args.param2, 
                      bench=args.bench, ngram=args.ngram, temperature=args.temperature, wm=args.wm,gen_len=args.gen_len,
                      beam_chunk_size=args.beam_chunk_size)

    if args.robust:
        config.mode = f"robust-{config.wm}"
    fpath = generate_json_filenames(config, 'all_scores_', suffix='', ext='.pkl')
    outpath = path.join(res_path, fpath)
    print("Saving all scores to:", outpath)


    tokenizer,_,_ = config_model(args.model_name, False)
    detector, scoring_method = get_detector(tokenizer, config,robust=args.robust)

    if args.robust:
        config.mode = config.wm
    scores = compute_all_scores(detector,scoring_method, config,res_path,outpath)


    with open(outpath, 'wb') as f:
         pickle.dump(scores, f)
    return(0)
if __name__ == '__main__':
    main()
