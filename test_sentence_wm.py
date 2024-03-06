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
from misc.wm_dataclasses import *

# To reproduce the data from the paper on MMW and C4 use:
# python test_sentence_wm.py  --generate --detect --seed=815 --ngram=4 --n=14 --N=64 --batch_size=1 --benches story_reports  fake_news invented_stories c4                  
#python test_sentence_wm.py  --mode max-inf-kirch --generate --detect --seed=1015 --ngram=4 --param1=0.95 --param2=0.15 --batch_size=10 --benches story_reports  fake_news invented_stories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--mode', type=str, default='sentence-wm')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ngram', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--outputdir', type=str, default='results/benchmark/')
    parser.add_argument('--standard_outputpath', type=str, default='./MarkMyWords/run/benchmark/')

    parser.add_argument('--param1','--n', '--delta', '--theta', type=int_or_float, default=3) #Number of texts generated per splits
    parser.add_argument('--param2','--N', '--gamma', type=opt_int_or_float, default=None) #Number of splits
    parser.add_argument('--temperature', type=float, default=1.) #Temperature

    parser.add_argument('--gen_len', type=int, default=1024) #Size of the generated text
    parser.add_argument('--beam_search', action=argparse.BooleanOptionalAction) # Only compatible with WaterMax
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benches', nargs='+', type=str, default='story_reports')

    parser.add_argument('--generate', action=argparse.BooleanOptionalAction)
   

    parser.add_argument('--detect', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rate', action=argparse.BooleanOptionalAction)

    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--standardize_final', action=argparse.BooleanOptionalAction)
    parser.add_argument('--detect_std', action=argparse.BooleanOptionalAction)
    parser.add_argument('--detect_robust', action=argparse.BooleanOptionalAction)
    parser.add_argument('--detect_std_robust', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fp32', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction)




    return parser


##HELPER
def get_generator(model, tokenizer, args):
    if args.mode == 'sentence-wm':
        generator = RobustWmSentenceGenerator(model, tokenizer, ngram=args.ngram, num_seq=args.param1, seed=args.seed)
    elif args.mode == 'nowm':
        generator = WmGenerator(model, tokenizer, ngram=args.ngram, seed=args.seed)
    elif args.mode == 'kirch':
        generator = MarylandGenerator(model, tokenizer, ngram=args.ngram, delta=args.param1, gamma=args.param2,seed=args.seed)
    elif args.mode == 'aaronson':
        generator = OpenaiGenerator(model, tokenizer, ngram=args.ngram, theta=args.param1,seed=args.seed)
    else:
        raise Exception("Unimplemented watermarking scheme!")
    return(generator)
def get_detector(tokenizer, args):
    if args.mode == 'sentence-wm':
        detector = GaussianSentenceWm(tokenizer, ngram=args.ngram, seed=args.seed,split_len=args.gen_len//args.param2)
    elif args.mode == 'kirch':
        detector= MarylandDetector(tokenizer, ngram=args.ngram, delta=args.param1, gam=args.param2, seed=args.seed)
    elif args.mode == 'aaronson':
        detector = OpenaiDetector(tokenizer, ngram=args.ngram, seed=args.seed)

    else:
        raise Exception("Unimplemented watermarking scheme!")
    return(detector)

def generate_mmw_path(args):
    if (args.mode == 'nowm') or (args.param1 is None and args.param2 is None) or (args.param2 == 1 and args.param1 == 1) or (args.param1 == 0):
        return(f"{args.seed}_{args.gen_len}_{args.bench}_{args.ngram}_{args.temperature}_nowm")

    elif args.param2 is None:
        return(f"{args.seed}_{args.gen_len}_{args.bench}_{args.ngram}_{args.temperature}_{args.param1}")
    else:
        return(f"{args.seed}_{args.gen_len}_{args.bench}_{args.ngram}_{args.temperature}_{args.param1}_{args.param2}")
    
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
        with open(jsondir, "r") as f:
            for _ in f:
                start_point += 1

    print(f"Starting from {start_point}")
    all_times = []
    with open(jsondir, "a") as f:
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
                f.write(json.dumps({
                    "prompt": prompt,
                    "result": result[len(prompt)+prompt_offset:],
                    "speed": speed,
                    "eta": eta}) + "\n")
                f.flush()

    print(f"Average time per prompt: {np.sum(all_times) / (len(prompts) - start_point) :.2f}")
def generate(generator, prompts, jsondir, gen_len,  batch_size=1,temperature=1.,prompt_offset=1,top_p=0.95):
    """
    gen_len: total size of the text
    """
    start_point = 0

    #if path.isfile(jsondir): os.remove(jsondir)
    
    if path.exists(jsondir):
        with open(jsondir, "r") as f:
            for _ in f:
                start_point += 1

    print(f"Starting from {start_point}")
    all_times = []
    with open(jsondir, "a") as f:
        for ii in range(start_point, len(prompts), batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(batch_size, len(prompts) - ii)
            results = generator.generate(
                    prompts[ii:ii+chunk_size], 
                    max_gen_len=gen_len, 
                    temperature=temperature, 
                    top_p=top_p
                )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                #print("logging")
                f.write(json.dumps({
                    "prompt": prompt,
                    "result": result[len(prompt)+prompt_offset:],
                    "speed": speed,
                    "eta": eta}) + "\n")
                    #"deflection": float(generator.deflection)}) + "\n")
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


def detect(detector,results,jsondir_out):
    log_stats = []
    text_index = 0
    if path.isfile(jsondir_out): remove(jsondir_out)
    with open(jsondir_out, 'w') as f:
        for text in tqdm.tqdm(results):#tqdm.tqdm(zip(results, results_orig)):
            # compute watermark score
            scores_no_aggreg = detector.get_scores_by_t([text], scoring_method='v2')
            scores = detector.aggregate_scores(scores_no_aggreg, aggregation='sum') # p 1
            pvalues = detector.get_pvalues(scores_no_aggreg) 

            #payloads = [ 0 ] * len(pvalues)
            if  pvalues.size >0:
                pvalues = pvalues[:,0].tolist()
                all_pvalues = pvalues
                try:
                    scores = [float(s[0]) for s in scores]
                except:
                    scores = [-1]

                # log stats and write
                log_stat = {
                        'text_index': text_index,
                        'num_token': len(detector.tokenizer.encode(text,add_special_tokens=False)),
                        'score': scores[0],
                        'pvalue': pvalues[0], 
                        'all_pvalues': all_pvalues[0],
                }
                log_stats.append(log_stat)
                f.write(json.dumps(log_stat)+'\n')
            text_index += 1
        df = pd.DataFrame(log_stats)
        df['log10_pvalue'] = np.log10(df['pvalue'])
        print(f">>> Scores: \n{df.describe(percentiles=[])}")
        print(f"Saved scores to {jsondir_out}") 


def standardize_output(gen_jsondir, detect_jsondir, outfile, seed,watermark_config=None, watermarked=True,temperature=1.):
    gen_prompts = load_results(gen_jsondir, result_key="prompt")
    gen_results = load_results(gen_jsondir)
    detect_results = pd.DataFrame(load_scores(detect_jsondir))
    if temperature == 'beam_search': temperature=0.0
    gen_list = []


    for i in range(len(detect_results)):
        gen_data = {"id":i,
                "prompt": gen_prompts[i],
                "response":gen_results[i],
                "token_count":int(detect_results['num_token'][i]),
                "temp":temperature
                }
        if watermarked:
            gen_data['key'] = seed
            gen_data['watermark'] = DATA_SentenceWatermarkSpec.from_dict(watermark_config)
            gen_data["pvalue"] = detect_results['pvalue'][i]
        gen_list.append(DATA_Generation.from_dict(gen_data))
    DATA_Generation.to_file(outfile, gen_list)
    return(gen_list)
def standardize_output_final(std_res, detect_jsondir, outfile, seed,watermark_config=None,temperature=1.):

    detect_results = pd.DataFrame(load_scores(detect_jsondir))
    gen_list = []
    if temperature == 'beam_search': temperature = 0.0

    for i in range(len(detect_results)):
        gen_data = {"id":std_res[i].id,
                "prompt": std_res[i].prompt,
                "response":std_res[i].response,
                "token_count":int(detect_results['num_token'][i]),
                "attack": std_res[i].attack,
                "rating": std_res[i].rating,
                "temp":temperature,
                }
        if std_res[i].watermark is not None:
            gen_data['key'] = seed
            gen_data['watermark'] = DATA_SentenceWatermarkSpec.from_dict(watermark_config)
            gen_data["pvalue"] = detect_results['pvalue'][i]
        gen_list.append(DATA_Generation.from_dict(gen_data))
    DATA_Generation.to_file(outfile, gen_list)
    return(gen_list)
    
def rate(rater, prompts, jsondir, batch_size=10):
    start_point = 0

    
    if path.exists(jsondir):
        with open(jsondir, "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")
    all_times = []
    with open(jsondir, "a") as f:
        for ii in range(start_point, len(prompts), batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(batch_size, len(prompts) - ii)
            results = rater.rate(prompts[ii:ii+chunk_size])
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                f.write(json.dumps({
                    "prompt": prompt,
                    "rating": result,
                    "speed": speed,
                    "eta": eta}) + "\n")
                f.flush()

    print(f"Average time per prompt: {np.sum(all_times) / (len(prompts) - start_point) :.2f}")
    
    
def main(args,tokenizer,model):


    # File book-keeping
    model_ref = model_name.split('/')[-1]

    makedirs(path.join(args.outputdir, model_ref), exist_ok=True)
    generate_jsondir = path.join(args.outputdir, model_ref, generate_json_filenames(args, prefix='results_'))
    rate_jsondir = path.join(args.outputdir,  model_ref,  generate_json_filenames(args, prefix='rating_'))

    detect_jsondir = path.join(args.outputdir,  model_ref,  generate_json_filenames(args, prefix='scores_'))
    detect_jsondir_final = path.join(args.outputdir,  model_ref,  generate_json_filenames(args, prefix='final_scores_'))

    

    print("Saving generation in: ", generate_jsondir)
    print("Saving detection in: ", detect_jsondir)
    print("Saving final detection in: ", detect_jsondir_final)
    


    torch.manual_seed(args.seed) # Reset PRNG

    #GENERATION
    if args.generate:
        if not args.beam_search: print("Sampling mode")
        else: print("Tree search mode")
        prompts = generate_prompts(args.bench, model_name)
        generator = get_generator(model,tokenizer,args)
        prompt_offset = 1
        if args.bench == 'c4' : prompt_offset =0
        if args.mode == 'sentence-wm':
            watermax_generate(generator, prompts, generate_jsondir, args.param1,args.param2,args.gen_len, batch_size=args.batch_size, beam_search=args.beam_search,temperature=args.temperature,prompt_offset=prompt_offset)
        else:
            top_p = 0.95
            print("Nucleus: ", top_p)
            generate(generator, prompts, generate_jsondir, args.gen_len, batch_size=args.batch_size, temperature=args.temperature,prompt_offset=prompt_offset,top_p=top_p)

    #DETECTION
    if args.detect:
        detector = get_detector(tokenizer,args)
        results = load_results(generate_jsondir)
        if args.mode == 'sentence-wm' or args.mode == 'inf-watermax' or args.mode == 'inf-kirch' or args.mode == 'star-inf-kirch' or args.mode == 'max-inf-kirch' or args.mode == 'kirch-watermax':
            detect_watermax(detector,results,detect_jsondir)
        else: detect(detector,results,detect_jsondir)
    if  args.detect_robust:
        detect_jsondir = path.join(args.outputdir,  model_ref, generate_json_filenames(args, prefix='scores_robust_'))
        if args.mode == 'sentence-wm':
            detector = SecureGaussianSentenceWm(tokenizer, ngram=args.ngram, seed=args.seed)
        results = load_results(generate_jsondir)
        detect_watermax(detector,results,detect_jsondir,compute_score=False)


    #RATING
            
    if args.rate:
        rater = TextRater(model, tokenizer)
        qprompts = load_rating_prompt(generate_jsondir)
        rate(rater, qprompts, rate_jsondir, batch_size=10)
    
    """
    MMW specific options

    The rest of the options are for standardizing the results in order for them to be used within the MarkMyWords benchmark,
    notably the attack suite. The detection option at this point is for computing the pvalues of attacked texts only.

    """
    #STANDARDIZE (for MMW benchmark)
    if args.beam_search: args.temperature = 'beam_search' #Only impacts the filenames from that point on
    if args.standardize:

        standard_outputpath = path.join(args.standard_outputpath, args.mode, model_ref, generate_mmw_path(args))
        makedirs(standard_outputpath, exist_ok=True)
        standard_outfile = path.join(standard_outputpath , 'generations.tsv')
        print("Standard path:", standard_outfile)
        if args.param1 == 1 and args.param2 ==1:
            _=standardize_output(generate_jsondir, detect_jsondir, standard_outfile,args.seed, watermarked=False,watermark_config=None,temperature=args.temperature)
        else:
            watermark_config = {"generator": args.mode,
                    "tokenizer": model_name,
                    "temp": args.temperature,
                    "param1": args.param1,
                    "param2": args.param2,
                    "ngram":args.ngram,}
            _=standardize_output(generate_jsondir, detect_jsondir, standard_outfile,args.seed, watermarked=True,watermark_config=watermark_config,temperature=args.temperature)
    
    
    
    #DETECTION (for MMW benchmark)
    if args.detect_std:
        standard_outputpath = path.join(args.standard_outputpath, args.mode, model_ref, generate_mmw_path(args))

        standard_infile = path.join(standard_outputpath , 'perturbed.tsv')
        
        if not path.isfile(standard_infile) : raise FileNotFoundError(standard_infile)

        rated_gen = [d.response for d in DATA_Generation.from_file(standard_infile)]
        rated_prompts = [d.prompt for d in DATA_Generation.from_file(standard_infile)]
        detector = get_detector(tokenizer,args)
        if args.mode == 'sentence-wm':detect_watermax(detector,rated_gen,detect_jsondir_final)
        else: detect(detector,rated_gen,detect_jsondir_final)
    if args.mode == 'sentence-wm'  and args.detect_robust:
        standard_outputpath = path.join(args.standard_outputpath, args.mode, model_ref, generate_mmw_path(args))

        standard_infile = path.join(standard_outputpath , 'perturbed.tsv')
        
        if not path.isfile(standard_infile) : raise FileNotFoundError(standard_infile)
        detect_jsondir = path.join(args.outputdir,  model_ref, generate_json_filenames(args, prefix='scores_robust_final_'))

        rated_gen = [d.response for d in DATA_Generation.from_file(standard_infile)]
        detector = SecureGaussianSentenceWm(tokenizer, ngram=args.ngram, seed=args.seed)
        
        detect_watermax(detector,rated_gen,detect_jsondir_final,compute_score=False)
    
    #STANDARDIZE FINAL (for MMW benchmark)
    if args.standardize_final:
        standard_outputpath = path.join(args.standard_outputpath, args.mode, model_ref, generate_mmw_path(args))

        standard_outfile = path.join(standard_outputpath , 'detect.tsv')
        standard_infile = path.join(standard_outputpath , 'rated.tsv')
        if not path.isfile(standard_infile) : raise FileNotFoundError(standard_infile)
        if path.isfile(standard_outfile) : remove(standard_outfile)
        rated_std= DATA_Generation.from_file(standard_infile)

        watermark_config = {"generator": args.mode,
                "tokenizer": model_name,
                "temp": 1.0,
                "N": args.param2,
                "n": args.param1,
                "ngram":args.ngram,}
        d=standardize_output_final(rated_std, detect_jsondir_final, standard_outfile,args.seed,watermark_config=watermark_config)
        print(len(d))


if __name__ == '__main__':
    parser = parse_arguments()
    parser.add_argument('--bench', type=str,  default='No Bench') # HACK: Placeholder to loop on

    args = parser.parse_args()
    
    model_name = args.model_name
    

    if args.generate or args.detect or args.detect_std or args.detect_robust or args.detect_std_robust or args.rate:
        if args.fp32: dtype= torch.float32
        elif args.fp16: dtype = torch.float16
        else: dtype= torch.bfloat16
        tokenizer, model,prompt_type = config_model(model_name,args.generate or args.rate,dtype=dtype)
    else:
        tokenizer,model, = (None,None)

    print("Benchmarks: ", args.benches)
    for bench in args.benches:
        args.bench = bench
        main(args, tokenizer,model)
