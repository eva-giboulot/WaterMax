from os import path
import argparse

from misc.wm_dataclasses import WmConfig, int_or_float,opt_int_or_float
from misc.helpers import load_json, config_model, generate_json_filenames

from scores.other_metrics import ppl

import json

#python compute_ppl.py --wm=kirch --seed=1015 --ngram=4 --generator_name=[PATH_TO_MODEL]/Mistral-7B-Instruct-v0.2 --param1=5.0 --param2=0.5 --benches=fake_news
# python compute_ppl.py --wm=kirch --seed=1015 --ngram=4 --generator_name=[PATH_TO_MODEL]/Mistral-7B-Instruct-v0.2 --param1=1.0 --param2=0.5 --benches fake_news story_reports invented_stories
def parse_arguments():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--wm', type=str, default='kirch')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ngram', type=int, default=8)
    parser.add_argument('--generator_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--oracle_name', type=str, default='facebook/opt-2.7b')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--beam_chunk_size', type=int, default=0) 


    parser.add_argument('--res_path', type=str, default='./results/benchmark')


    parser.add_argument('--param1', type=int_or_float) 
    parser.add_argument('--param2', type=opt_int_or_float) 
    parser.add_argument('--temperature', type=float, default=1.) #Temperature

    parser.add_argument('--gen_len', type=int, default=1024) #Size of the generated text
    parser.add_argument('--robust', action=argparse.BooleanOptionalAction)
    parser.add_argument('--benches', nargs='+', type=str, default='story_reports')
    return(parser)

def compute_ppl(tokenizer, model, config,res_path,batch_size):
    txt_path = generate_json_filenames(config, prefix='results_', ext='.jsonl')
    texts = load_json(path.join(res_path, txt_path),key='result')
    prompts = load_json(path.join(res_path, txt_path),key='prompt')
    
    base_ppls = ppl(texts, model,tokenizer, batch_size= batch_size)

   
    return(prompts, texts, base_ppls)




def main():
    args = parse_arguments().parse_args()
    res_path = path.join(args.res_path,args.generator_name.split('/')[-1]) 
    tokenizer,model,_ = config_model(args.oracle_name, True)
    print(args.benches)
    for bench in args.benches:
        config = WmConfig(seed=args.seed, param1=args.param1, param2=args.param2, bench=bench, ngram=args.ngram,
                           temperature=args.temperature, wm=args.wm,gen_len=args.gen_len,beam_chunk_size=args.beam_chunk_size)
        prompts, texts, base_ppls = compute_ppl(tokenizer, model, config,res_path,batch_size=args.batch_size)
        
        jsondir =path.join(res_path, generate_json_filenames(config, prefix='results_ppl_', suffix='_' + args.oracle_name.split('/')[-1], ext='.jsonl'))
        #jsondir= path.join(res_path, f"results_ppl_{config.bench}_{config.seed}_{config.wm}_{config.param1}_{config.param2}_{config.gen_len}_{config.ngram}_{config.temperature}_{args.oracle_name.split('/')[-1]}.jsonl")
        
        with open(jsondir, "w") as f:
                for prompt, txt, base_ppl in zip(prompts, texts, base_ppls):
                    f.write(json.dumps({
                        "prompt": prompt,
                        "result": txt,
                        "ppl": base_ppl,
                        }) + "\n")
                    f.flush()

if __name__ == '__main__':
    main()
