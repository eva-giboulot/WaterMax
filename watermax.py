from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import argparse
from os import path,makedirs,remove
import json,time

import tqdm



from models.wm import *
from misc.helpers import *
from misc.wm_dataclasses import int_or_float,opt_int_or_float

# python watermax.py  --generate --detect --seed=815 --ngram=4 --n=2 --N=2 --prompts data/test_prompts
# python watermax.py  --generate --detect --seed=815 --ngram=4 --n=2 --N=2 --prompts "What was Spinoza's relationshyip with Leibniz ? " "Which philospher spoke about the multicolored cow?"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ngram', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--outputdir', type=str, default='results/interactive_prompting/')
    parser.add_argument('--param1','--n', type=int_or_float, default=3) #Number of texts generated per splits
    parser.add_argument('--param2','--N', type=opt_int_or_float, default=None) #Number of splits
    parser.add_argument('--temperature', type=float, default=1.) #Temperature
    parser.add_argument('--beam_search', action=argparse.BooleanOptionalAction) # Only compatible with WaterMax
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--gen_len', type=int, default=256) #Size of the generated text

    parser.add_argument('--generate', action=argparse.BooleanOptionalAction)
    parser.add_argument('--detect', action=argparse.BooleanOptionalAction)

    parser.add_argument('--detect_robust', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fp32', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction)
    parser.add_argument('--quantize', action=argparse.BooleanOptionalAction)


    parser.add_argument('--prompts', nargs='+', type=str, default='Tell me about Deleuze concept of body without organs.')




    return parser


    
## MAIN
def watermax_generate(generator, prompts, jsondir, n,N,gen_len, batch_size=1,beam_search=False,temperature=1., prompt_offset=1):
    """
    n: number of text generated per splits
    N: number of splits
    gen_len: total size of the text
    """
    print("Beam Search:", beam_search)
    start_point = 0
    if not beam_search: num_beams=1
    else: num_beams=n
    #if path.isfile(jsondir): os.remove(jsondir)
    
    if path.exists(jsondir):
        remove(jsondir)

    all_times = []
    with open(jsondir, "w") as f:
        for ii in range(start_point, len(prompts), batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(batch_size, len(prompts) - ii)
            results = generator.generate(
                prompts[ii:ii+chunk_size], 
                max_gen_len=gen_len, 
                temperature=temperature, 
                top_p=1.0,
                n_splits=N,
                do_sample=not beam_search,
                num_beams=num_beams)
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                print("Output: ", result[len(prompt)+prompt_offset:])
                f.write(json.dumps({
                    "prompt": prompt,
                    "result": result[len(prompt)+prompt_offset:],
                    "speed": speed,
                    "eta": eta}) + "\n")
                f.flush()

    print(f"Average time per prompt: {np.sum(all_times) / (len(prompts) - start_point) :.2f}")

def detect_watermax(detector,results,jsondir_out,compute_score=True):
    log_stats = []
    text_index = 0
    if path.isfile(jsondir_out): remove(jsondir_out)
    with open(jsondir_out, 'w') as f:
        for text in tqdm.tqdm(results):#tqdm.tqdm(zip(results, results_orig)):
            # compute watermark score
            scores_no_aggreg = detector.get_scores_by_t([text], scoring_method='v1')
            if compute_score: scores = detector.aggregate_scores(scores_no_aggreg, aggregation='sum') # p 1
            pvalues = np.array([detector.get_pvalues(scores_no_aggreg[0]) ])

            #payloads = [ 0 ] * len(pvalues)
            if  pvalues.size >0:
                pvalues = pvalues.tolist()
                all_pvalues = pvalues
                if compute_score : 
                    try:
                        scores = [float(s[0]) for s in scores]
                    except:
                        scores = [-1]
                #num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
                # compute sbert score
                #xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
                #score_sbert = cossim(xs[0], xs[1]).item()
                # log stats and write
                log_stat = {
                        'text_index': text_index,
                        'num_token': len(detector.tokenizer.encode(text,add_special_tokens=False)),
                        'score': scores[0] if compute_score else -1,
                        'pvalue': pvalues[0], 
                        'all_pvalues': all_pvalues[0],
                        #'score_sbert': score_sbert,
                        #'payload': payloads[0],
                }
                log_stats.append(log_stat)
                f.write(json.dumps(log_stat)+'\n')
            text_index += 1
        df = pd.DataFrame(log_stats)
        df['log10_pvalue'] = np.log10(df['pvalue'])
        print(f">>> Scores: \n{df.describe(percentiles=[])}")
        print(f"Saved scores to {jsondir_out}") 






    
    
def main(args,tokenizer,model,prompts):


    # File book-keeping
    model_ref = model_name.split('/')[-1]

    makedirs(path.join(args.outputdir, model_ref), exist_ok=True)
    generate_jsondir = path.join(args.outputdir, model_ref, generate_json_filenames(args, prefix='results_'))
    detect_jsondir = path.join(args.outputdir,  model_ref,  generate_json_filenames(args, prefix='scores_'))


    

    print("Saving generation in: ", generate_jsondir)
    print("Saving detection in: ", detect_jsondir)

    


    torch.manual_seed(args.seed) # Reset PRNG

    #GENERATION
    if args.generate:
        if not args.beam_search: print("Sampling mode")
        else: print("Tree search mode")
        generator = RobustWmSentenceGenerator(model, tokenizer, ngram=args.ngram, num_seq=args.param1, seed=args.seed)
        prompt_offset = 1
        watermax_generate(generator, prompts, generate_jsondir, args.param1,args.param2,args.gen_len, batch_size=args.batch_size, beam_search=args.beam_search,temperature=args.temperature,prompt_offset=prompt_offset)

    #DETECTION
    if args.detect:
        detector = GaussianSentenceWm(tokenizer, ngram=args.ngram, seed=args.seed,split_len=args.gen_len//args.param2)
        results = load_results(generate_jsondir)
        detect_watermax(detector,results,detect_jsondir)
    if  args.detect_robust:
        detect_jsondir = path.join(args.outputdir,  model_ref, generate_json_filenames(args, prefix='scores_robust_'))
        detector = SecureGaussianSentenceWm(tokenizer, ngram=args.ngram, seed=args.seed)
        results = load_results(generate_jsondir)
        detect_watermax(detector,results,detect_jsondir,compute_score=False)


    


if __name__ == '__main__':
    parser = parse_arguments()
    parser.add_argument('--mode', type=str) # HACK: Make compatible with test-sentence-wm helpers
    sys_prompt = "You are a helpful assistant. Always respond truthfully and to the best of your ability."
    args = parser.parse_args()

    args.mode = 'sentence-wm'
    model_name = args.model_name
    
    if args.prompts[0].endswith(".txt"):
        print("Loading prompts from txt file")
        with open(args.prompts[0], 'r') as f:
            prompts = [standardize(model_name.split('/')[-1], sys_prompt, user_prompt) for user_prompt in f] 
            print(prompts)
    else:
        prompts=[standardize(model_name.split('/')[-1], sys_prompt, user_prompt) for user_prompt in args.prompts]
        print("Prompt: ", prompts)

    if args.fp32: dtype= torch.float32
    elif args.fp16: dtype = torch.float16
    else: dtype= torch.bfloat16
    tokenizer, model,_ = config_model(model_name,args.generate or args.rate,dtype=dtype, quantize=args.quantize)



    main(args, tokenizer,model,prompts)
