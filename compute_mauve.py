from os import path
import argparse

from misc.wm_dataclasses import WmConfig, int_or_float,opt_int_or_float
from misc.helpers import load_json,config_model, generate_json_filenames

from evaluate import load


import pickle

#python compute_mauve.py --wm kirch -- seed 1015 --ngram 4 --model_name [PATH_TO_MODEL]/Mistral-7B-Instruct-v0.2 --param1 5.0 --param2 0.5 --benches fake_news story_reports invented_stories

def parse_arguments():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--wm', type=str, default='kirch')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ngram', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')

    parser.add_argument('--res_path', type=str, default='./results/benchmark')


    parser.add_argument('--param1', type=int_or_float,default=1) 
    parser.add_argument('--param2', type=opt_int_or_float) 
    parser.add_argument('--temperature', type=float, default=1.) #Temperature

    parser.add_argument('--robust', action=argparse.BooleanOptionalAction)
    parser.add_argument('--benches', nargs='+', type=str, default='story_reports')
    parser.add_argument('--gen_len', type=int, default=1024) #Size of the generated text

    return(parser)

def compute_mauve(config, res_path):
    mauve = load('mauve')
    txt_path = generate_json_filenames(config, prefix='results_', ext='.jsonl')
    wm_texts = load_json(path.join(res_path, txt_path),key='result')


    if config.wm == 'sentence-wm': 
         #Base WaterMax direcly uses huggingface generator contrary
         #to the other wm schemes which use Meta's implementation of sampling
         config.param1=1
         config.param2=1
    else:
        config.wm = 'nowm'
        config.mode = 'nowm'


    clean_txt_path = generate_json_filenames(config, prefix='results_', ext='.jsonl')

    try:
        texts_clean = load_json(path.join(res_path, clean_txt_path),key='result')
    except:
         FileNotFoundError("Non-watermarked text for {}/{} has not been generated: {}".format(config.wm, 
                                                                                              config.bench,
                                                                                              path.join(res_path, clean_txt_path)))

    

    mauve_results = mauve.compute(predictions=wm_texts, references=texts_clean)

   
    return(mauve_results)




def main():
    args = parse_arguments().parse_args()
    res_path = path.join(args.res_path,args.model_name.split('/')[-1]) 
    print(args.benches)
    for bench in args.benches:
        config = WmConfig(seed=args.seed, param1=args.param1, param2=args.param2, bench=bench, ngram=args.ngram, temperature=args.temperature, wm=args.wm,gen_len=args.gen_len)
        mauve_results=compute_mauve(config, res_path)
        config.wm = args.wm
        config.mode = args.wm

        if args.wm == 'sentence-wm':
            config.param1=args.param1
            config.param2=args.param2

        print(f"MAUVE {config.wm}/{config.param1}: {mauve_results.mauve}")
        resdir =path.join(res_path, generate_json_filenames(config, prefix='mauve_', ext='.pkl'))
        print("Saving in:", resdir)
        #jsondir= path.join(res_path, f"results_ppl_{config.bench}_{config.seed}_{config.wm}_{config.param1}_{config.param2}_{config.gen_len}_{config.ngram}_{config.temperature}_{args.oracle_name.split('/')[-1]}.jsonl")
        with open(resdir, "wb") as f:
                pickle.dump(mauve_results,f)
if __name__ == '__main__':
    main()
