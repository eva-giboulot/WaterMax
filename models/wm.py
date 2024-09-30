import torch
from transformers import AutoTokenizer,LlamaForCausalLM, LlamaTokenizer
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import numpy.random as npr
from scipy.stats import norm, gamma,binom
from scipy import special
#from misc.power_math import power_cdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Base classes from Three Bricks
"""
class WmDetector():
    def __init__(self, 
            tokenizer: AutoTokenizer, 
                ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size #len(self.tokenizer) #self.tokenizer.vocab_size
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores)
        if aggregation == 'sum':
           return [ss.sum(axis=0) for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else np.ones(shape=(self.vocab_size)) for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) for ss in scores]
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt.numpy()[:payload_max+1]
                rts.append(rt)
            score_lists.append(rts)
        return score_lists

    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x ntoks x payload_max
        for ss in scores:
            ntoks = ss.shape[0]
            scores_by_payload = ss.sum(axis=0) if ntoks!=0 else np.zeros(shape=ss.shape[-1]) # payload_max
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in scores_by_payload]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max

    def get_pvalues_by_t(self, scores: List[float]) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError

class WmGenerator():
    def __init__(self, 
            model: LlamaForCausalLM, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0
        ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = model.config.max_sequence_length
        self.pad_id = model.config.pad_token_id
        self.eos_id = model.config.eos_token_id
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token



"""
WaterMax classes

"""
class RobustWmSentenceGenerator():
    
    def __init__(self, 
            model, 
            tokenizer,
            seed :int = 0,
            seeding:str = 'hash',
            salt_key:int = 35317,
            ngram:int = 3,
            num_seq:int=10,
            eos_value = np.inf, # How to score outputs containing a single eos token, np.nan will skip them, while np.inf will stop the generation
                ): 
            
        self.model = model
        self.tokenizer = tokenizer
        self.salt_key = salt_key
        self.seed = seed
        self.rng = npr.default_rng(seed)
        self.ngram = ngram
        self.seeding = seeding
        self.num_seq = num_seq
        self.scores = []
        self.generate_counter =0
        self.hash_size = (2 ** 64 - 1)
        self.eos_value = eos_value
        if self.ngram ==0:
            self.change_rng_seed(self.seed)
            self.fixed_u = self.rng.standard_normal(len(tokenizer))


        else: self.fixed_u = None
        self.seen_ngram = []
    
    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    def change_rng_seed(self, seed): #BEWARE: Impure function based on side-effect
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state
    
    def get_seed_rng(
        self, 
        input_ids
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for jj,i in enumerate(input_ids):
                seed = int(np.mod(seed * self.salt_key + i, self.hash_size))
                #print(jj,i,seed)
        elif self.seeding == 'additive':
            seed = self.salt_key * np.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = np.min(seed)
        return seed
    
    def scoring_outputs(self, input_len, outputs, chaining_ngrams=True):
        scores = np.zeros(len(outputs))
        if self.ngram > 0:
            us_tmp = []
            for ii in range(len(outputs)):
                us_tmp.append([])
                input_offset = input_len[ii//self.num_seq] #len(inputs['input_ids'][ii//self.num_seq])
                if len(outputs[ii][input_offset:]) != 0:
                    Xagg = 0
                    for jj in range(input_offset, len(outputs[ii])): #Offset by prompt from tokenizer to synchronize properly
                        if not chaining_ngrams and (jj < input_offset + self.ngram -1): 
                            ngram_w = outputs[ii][input_offset:][-self.ngram+(jj-input_offset)+1:] + outputs[ii][input_offset:][:(jj-input_offset)+1]
                        else:
                            ngram_w = outputs[ii][jj-self.ngram+1:jj+1]
                        if tuple(ngram_w) in self.seen_ngram[ii]:
                            continue
                        else:
                            seed = self.get_seed_rng(ngram_w)
                            self.change_rng_seed(seed) #SIDE EFFECT: changes rng state, reset to base state of the seed
                            u = self.rng.standard_normal()
                            us_tmp[ii].append(u)
                            Xagg += u # Allows a Gaussian hypo under H0 and Gumbel-type (EVD) under H1

                            self.seen_ngram[ii].add(tuple(ngram_w))
                    if len(us_tmp[ii]) > 0:
                        scores[ii] = Xagg/np.sqrt(len(us_tmp[ii]))
                    else: scores[ii] = -np.inf # No scorable ngrams, reject text
                else:
                    scores[ii] = self.eos_value # Output is empty, meaning it only contained special characters (i.e <eos>)
                 
                
        elif self.ngram == 0:
            for ii in range(len(outputs)):
                input_offset = input_len[ii//self.num_seq]
                if len(outputs[ii][input_offset:]) != 0:
                    #us_tmp = self.fixed_u[outputs[ii][input_offset:]]
                    us_tmp = []
                    for jj in range(input_offset, len(outputs[ii])):
                        ngram_w = outputs[ii][jj]
                        if tuple([ngram_w]) in self.seen_ngram[ii]:
                            continue
                        else:
                            us_tmp.append(self.fixed_u[ngram_w])
                            self.seen_ngram[ii].add(tuple([ngram_w]))

                    #print( self.tokenizer.decode(outputs[ii][input_offset:],skip_special_tokens=True))
                    if len(us_tmp) > 0:
                        Xagg = np.sum(us_tmp)/np.sqrt(len(us_tmp)) #Normalizing so that the aggregated score is a centered gaussian
                    else:
                        Xagg = -np.inf

                else:
                    Xagg = self.eos_value
                scores[ii] = Xagg 
        elif self.ngram==-1:
            for ii in range(len(outputs)):
                input_offset = input_len[ii//self.num_seq] #len(inputs['input_ids'][ii//self.num_seq])
                ngram_tokens = outputs[ii][input_offset:]
                if len(ngram_tokens ) != 0: # If we have not already reached <eos>
                    us_tmp = np.zeros(len(ngram_tokens ))
                    seed = self.get_seed_rng(ngram_tokens)
                    #We add more randomness in case of repeating tokens
                    for i in range(len(ngram_tokens)):
                        self.change_rng_seed(seed+i) #SIDE EFFECT: changes rng state, reset to base state of the seed
                        us_tmp[i] = self.rng.standard_normal()

                
                
                    Xagg = np.sum(us_tmp)/np.sqrt(us_tmp.size) #Normalizing so that the aggregated score is a centered gaussian
                else:
                    Xagg = self.eos_value
                scores[ii] = Xagg 
        return(scores)
    
    
    def contains_eos(self, t) -> bool:
        #print(t)
        if self.tokenizer.eos_token_id in t:
            return(True)
        else:
            return(False)
        
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        top_p: float = 0.95,
        do_sample:bool =True,
        num_beams: int =1,
        temperature:float =0.85,
        n_splits:int = 1,
        #num_return_sequences: int = 5
    ) -> List[str]: 
        split_len = max_gen_len//n_splits
        finished_text = np.zeros(len(prompts),dtype=bool)
        res = [[] for _ in range(len(prompts))]
        self.seen_ngram = [set() for _ in range(len(prompts)*self.num_seq)]
        for k in range(n_splits):
            if k == 0:
                inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to(device)
                
            else:
                inputs = self.tokenizer(res, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to(device)
                
            
            input_len = np.array([len(np.array(p)[np.argwhere(np.array(p) != self.tokenizer.eos_token_id)[0][0]:]) for p in inputs['input_ids'].cpu()])
            
            #print(np.repeat(np.arange(0,len(prompts)),self.num_seq))
            outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=num_beams, num_return_sequences=self.num_seq,max_new_tokens=split_len)
            is_eos_split = np.array(list(map(lambda x,y : self.contains_eos(x[input_len[y]:]), outputs,np.repeat(np.arange(0,len(prompts)),self.num_seq))))
            #outputs_sizes = np.array(list(map(lambda x,y : len(x[input_len[y]:]), outputs,np.repeat(np.arange(0,len(prompts)),self.num_seq))))

            # We need to decode re-encode in order to prevent some double tokens (such as \n\n) to be read as one at the detector side
            # Actually, we can't do much if the tokenizer adds or substract a token when decoding/recoding except reject the text
            # For Llama2, beware of things such as function name e.g factorial(num) will be outputed as a single token factorial( but recoded as two tokens factorial( 
            
            final_outputs= self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
           
            outputs = self.tokenizer(final_outputs,add_special_tokens=False)['input_ids']

            #if k==0: input_len -= 1 # Remove the starting <s> which was striped by skipping special_tokens 
            renc_outputs_sizes = np.array(list(map(lambda x,y : len(x[input_len[y]:]), outputs,np.repeat(np.arange(0,len(prompts)),self.num_seq))))
            
            if k==0:
                scores = self.scoring_outputs(input_len, outputs, chaining_ngrams=False) #We cannot base the seed on the prompt
            else:
                scores = self.scoring_outputs(input_len, outputs)

            valid_split_mask = (is_eos_split) | (split_len == renc_outputs_sizes) #If the split contains an eos, we can select it as the final split of the text
            if k < n_splits-1: scores[~valid_split_mask] = -np.inf # We reject pathological re-encodings, but not for the last split since it won't desynchronize the decoder
            scores = scores.reshape(len(prompts), self.num_seq, -1)
            is_eos_split = is_eos_split.reshape(len(prompts), self.num_seq)
            idxs = np.nanargmax(scores,axis=1) #Will raise an exception if all NaN !
            

            for ii,idx in enumerate(idxs):
                #print(idx)
                if not finished_text[ii]:
                    for jj in range(self.num_seq):
                        if jj != idx[0]: self.seen_ngram[ii*self.num_seq + jj] = self.seen_ngram[ii*self.num_seq + idx[0]].copy()
                        
                    res[ii] = final_outputs[ii*self.num_seq:(ii+1)*self.num_seq][idx[0]]
                    if is_eos_split[ii][idx[0]]: 
                        #print(f"Finished text: {ii}")
                        finished_text[ii] = True


        self.generate_counter +=1
        return (res)

class GaussianSentenceWm(WmDetector):
    def __init__(self, 
            tokenizer,
            split_len: int,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.rng = npr.default_rng(seed)
        self.us = []
        if self.ngram ==0:
            self.change_rng_seed(self.seed)
            self.fixed_u = self.rng.standard_normal(len(tokenizer))
            self.fixed_u[tokenizer.eos_token_id] = np.nan
            self.fixed_u[tokenizer.unk_token_id] = np.nan
        else: self.fixed_u = None
        self.split_len = split_len
        self.hash_size=2**64 -1
        self.ngram = ngram
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = int(np.mod(seed * self.salt_key + i, self.hash_size))
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed
    def change_rng_seed(self, seed): #BEWARE: Impure function based on side-effect
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state
    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x,add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = 0
            rts = []
            seen_ntuples = set()
            if self.ngram > 0:
                n_splits = total_len//self.split_len
                for cur_split in range(n_splits+1):
                    
                    cur_split_tokens_id = tokens_id[ii][self.split_len*cur_split:self.split_len*(cur_split+1)]
                    cur_split_size = np.min([self.split_len, len(cur_split_tokens_id)])
                    if cur_split_size >0:
                        rt = [] #np.zeros(cur_split_size)
                        #start_pos = self.split_len*cur_split
                        for cur_pos in range(start_pos, cur_split_size):
                            if (cur_pos < self.ngram-1):
                                if (cur_split ==0):
                                    ngram_tokens = cur_split_tokens_id[-self.ngram+cur_pos+1:] + cur_split_tokens_id[:cur_pos+1] 
                                else:
                                    prev_split_tokens_id = tokens_id[ii][self.split_len*(cur_split-1):self.split_len*cur_split]
                                    ngram_tokens = prev_split_tokens_id[-self.ngram+cur_pos+1:] + cur_split_tokens_id[:cur_pos+1]

                            else: ngram_tokens = cur_split_tokens_id[cur_pos-self.ngram+1:cur_pos+1]
                            if scoring_method == 'none':
                                 rt.append(self.score_tok(ngram_tokens))
                            elif scoring_method == 'v1':
                                #tup_for_unique = self.get_seed_rng(ngram_tokens)
                                tup_for_unique = tuple(ngram_tokens)
                                if not tup_for_unique in seen_ntuples:
                                    seen_ntuples.add(tup_for_unique)
                                    rt.append(self.score_tok(ngram_tokens))
                            #rt = rt.numpy()[:payload_max+1]
                        self.us.append(rt)
                        rts.append([np.nansum(rt)/np.sqrt(len(rt))])

                    

            elif self.ngram == 0:
                n_splits = total_len//self.split_len
                for cur_split in range(n_splits+1):
                    ngram_tokens = tokens_id[ii][self.split_len*cur_split:self.split_len*(cur_split+1)]
                    if len(ngram_tokens) > 0:
                        if scoring_method == 'none':
                            scores = [self.score_tok_ngram0(ngram_tokens)/np.sqrt(len(ngram_tokens))]
                        elif scoring_method == 'v1':
                            scores = []
                            for t in ngram_tokens:
                                tup_for_unique = tuple([t])
                                if not tup_for_unique in seen_ntuples:
                                    seen_ntuples.add(tup_for_unique)
                                    scores.append(self.fixed_u[t])
                                    self.us.append(self.fixed_u[t])
                            scores = [np.nansum(scores)/np.sqrt(len(scores))]
                                
                        rts.append(scores)
            elif self.ngram == -1:
                
                n_splits = total_len//self.split_len
                for cur_split in range(n_splits+1):
                    ngram_tokens = tokens_id[ii][self.split_len*cur_split:self.split_len*(cur_split+1)]
                    if len(ngram_tokens) > 0:
                        scores = [self.score_tok_ngrammax(ngram_tokens)]
                        rts.append(scores)
            else:
                raise Exception("Unimplemented ngram mode")
            #print(rts)
            score_lists.append(rts)
        return score_lists
    
    def score_tok_ngram0(self,ngram_tokens):
        scores = np.nansum(self.fixed_u[ngram_tokens])
        self.us.append(self.fixed_u[ngram_tokens])
        return scores
    def score_tok_ngrammax(self, ngram_tokens):
        """ 

        """
        #self.rng.manual_seed(seed)
        #print(ngram_tokens)
        seed = self.get_seed_rng(ngram_tokens)
        
        u = np.zeros(len(ngram_tokens))
        for i in range(len(ngram_tokens)):
            self.change_rng_seed(seed+i) #SIDE EFFECT: changes rng state, reset to base state of the seed
            u[i] = self.rng.standard_normal()
        #print(ngram_tokens,seed)
        scores = np.nansum(u)/np.sqrt(len(ngram_tokens))
        self.us.append(u)
        return scores
    def score_tok(self, ngram_tokens):
        """ 

        """
        seed = self.get_seed_rng(ngram_tokens)
        self.change_rng_seed(seed) #SIDE EFFECT: changes rng state, reset to base state of the seed
        u = self.rng.standard_normal()
        #print(ngram_tokens,seed)
        
        return u
    
    def get_pvalues(self, scores: float,  eps: float = 1e-200):
        """ from cdf of a gamma distribution """
        #if score < 0: pvalue = 1
        #else: pvalue = np.exp(-score**2/(2*ntoks))
        #print(np.sum(~np.isnan(scores)))
        pvalue = gamma.cdf(-np.nansum(norm.logcdf(scores)),a=np.sum(~np.isnan(scores)))
        return max(pvalue, eps)

class SecureGaussianSentenceWm(WmDetector):
    def __init__(self, 
            tokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.rng = npr.default_rng(seed)
        self.us = []
        if self.ngram ==0:
            self.change_rng_seed(self.seed)
            self.fixed_u = self.rng.standard_normal(len(tokenizer))
            self.fixed_u[tokenizer.eos_token_id] = np.nan
            self.fixed_u[tokenizer.unk_token_id] = np.nan
        else: self.fixed_u = None
        self.hash_size=2**64 -1
        self.ngram = ngram
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = int(np.mod(seed * self.salt_key + i, self.hash_size))
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed
    def change_rng_seed(self, seed): #BEWARE: Impure function based on side-effect
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state
    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x,add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = 0
            rts = []
            seen_ntuples = set()
            if self.ngram > 0:
                
                cur_split=0
                cur_split_tokens_id = tokens_id[ii][:]
                cur_split_size = len(cur_split_tokens_id)
                if cur_split_size >0:
                    rt = []
                    for cur_pos in range(start_pos, cur_split_size):
                        if (cur_pos < self.ngram-1):
                            if (cur_split ==0):
                                ngram_tokens = cur_split_tokens_id[-self.ngram+cur_pos+1:] + cur_split_tokens_id[:cur_pos+1] 
                            else:
                                prev_split_tokens_id = tokens_id[ii][self.split_len*(cur_split-1):self.split_len*cur_split]
                                ngram_tokens = prev_split_tokens_id[-self.ngram+cur_pos+1:] + cur_split_tokens_id[:cur_pos+1]

                        else: ngram_tokens = cur_split_tokens_id[cur_pos-self.ngram+1:cur_pos+1]
                        if scoring_method == 'none':
                             rt.append(self.score_tok(ngram_tokens))
                        elif scoring_method == 'v1':
                            #tup_for_unique = self.get_seed_rng(ngram_tokens)
                            tup_for_unique = tuple(ngram_tokens)
                            if not tup_for_unique in seen_ntuples:
                                seen_ntuples.add(tup_for_unique)
                                rt.append(self.score_tok(ngram_tokens))
                        #rt = rt.numpy()[:payload_max+1]
                    #self.us.append(rt)
                    #rts.append([np.sum(rt)/np.sqrt(len(rt))])
                    score_lists.append(rt)
        return score_lists
    

    def score_tok(self, ngram_tokens):
        """ 

        """
        seed = self.get_seed_rng(ngram_tokens)
        self.change_rng_seed(seed) #SIDE EFFECT: changes rng state, reset to base state of the seed
        u = self.rng.standard_normal()
        #print(ngram_tokens,seed)
        
        return u
    
    def get_pvalues(self, scores: float,  eps: float = 1e-200):
        """ from cdf of a gamma distribution """
        #if score < 0: pvalue = 1
        #else: pvalue = np.exp(-score**2/(2*ntoks))
        #print(np.sum(~np.isnan(scores)))
        pvalue = norm.sf(np.sum(scores)/np.sqrt(np.array(scores).size))
        return max(pvalue, eps)

class TextRater():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def check_rating(self,rating):
        rate = list(map(lambda x: x.split('/100')[0], rating ))
        for i,r in enumerate(rate):
           rate[i] = r if r.isnumeric() else None
        return(rate)
                
    def rate(self,prompts):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to(device)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=7,temperature=None, top_p=None)#['input_ids']
        res = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        rating = self.check_rating(res)
        return(rating)

class GroupAaronson(WmGenerator):
    """ Generate text using LLaMA and Aaronson's watermarking method. """
    def __init__(self, *args, theta=0.8, gam=2,**kwargs):
        super().__init__(*args, **kwargs)        
        self.theta = theta
        self.vocab_size = len(self.tokenizer)
        #self.dist = {'tv' : []}
        self.gamma = gam
        #self.gidx = []
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to generate V random number r between [0,1]
        - select argmax ( r^(1/p) )
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        temperature = self.theta
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            biased_probits = torch.zeros((ngram_tokens.shape[0], int(self.gamma)))



            for ii in range(ngram_tokens.shape[0]): # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                #print(seed)
                self.rng.manual_seed(seed)
                vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)#[probs_idx[ii].to('cpu')]
                pG = torch.zeros(int(self.gamma))

                #print(vocab_permutation[0:10])
                greenlist_groups =torch.zeros(self.vocab_size).to(probs_sort.device)
                for jj in range(int(self.gamma)):
                    greenlist = vocab_permutation[jj*int(1/self.gamma * self.vocab_size):(jj+1)*int(1/self.gamma * self.vocab_size)] # gamma * n
                    greenlist_groups[greenlist] = jj
                greenlist_groups = greenlist_groups[probs_idx[ii]]
                for jj in range(int(self.gamma)):
                    pG[jj] = torch.sum(probs_sort[ii,greenlist_groups == jj])

                # generate rs randomly between [0,1]
                rs = torch.rand(self.gamma, generator=self.rng) # n
                rs = torch.Tensor(rs).to(probs_sort.device)
                pG = torch.Tensor(pG).to(probs_sort.device)
                #print(pG)
                # compute r^(1/p)
                biased_probits[ii] = torch.pow(rs, 1/pG)
                #print(biased_probits)
                next_group = torch.argmax(biased_probits[ii])
                #self.gidx.append(next_group.item())
                probs_sort[ii, greenlist_groups != next_group] = 0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            #print(probs_sort)
            # select argmax ( r^(1/p) )
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
            #self.dist['tv'].append(1-probs_sort[:, next_token])

        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class GroupAaronsonDetector(WmDetector):

    def __init__(self, 
            tokenizer, 
            ngram: int = 1,
            seed: int = 0,
            gam: int = 2, 
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gam
        #self.gidx = []
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        #print(seed)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)

        rs = torch.rand(self.gamma, generator=self.rng) # n
        #print(vocab_permutation[0:5])
        greenlist_groups =torch.zeros(self.vocab_size)
        for jj in range(int(self.gamma)):
            greenlist = vocab_permutation[jj*int(1/self.gamma * self.vocab_size):(jj+1)*int(1/self.gamma * self.vocab_size)] # gamma * n
            greenlist_groups[greenlist] = jj
        group_id = int(greenlist_groups[token_id].item())
        #self.gidx.append(group_id)
        #print(group_id)
        scores = -torch.log(1 - rs[group_id:group_id+1])
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)

"""
Other SOTA, from Three Bricks
"""

class MarylandDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gam: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gam
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class OpenaiDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)


class OpenaiGenerator(WmGenerator):
    """ Generate text using LLaMA and Aaronson's watermarking method. """
    def __init__(self, *args, theta=0.8, **kwargs):
        super().__init__(*args, **kwargs)        
        self.theta = theta

        self.dist = {'tv' : []}
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to generate V random number r between [0,1]
        - select argmax ( r^(1/p) )
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        temperature = self.theta
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)[:, :self.tokenizer.vocab_size] #Need to remove the "added_tokens" in recent Phi3 implementations
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            biased_probits = torch.zeros_like(probs_sort)
            for ii in range(ngram_tokens.shape[0]): # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                rs = torch.rand(self.tokenizer.vocab_size, generator=self.rng) # n
                rs = rs.roll(-self.payload)
                rs = torch.Tensor(rs).to(probs_sort.device)
                rs = rs[probs_idx[ii]] 
                # compute r^(1/p)
                biased_probits[ii] = torch.pow(rs, 1/probs_sort[ii])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(biased_probits, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
            #self.dist['tv'].append(1-probs_sort[:, next_token])

        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
class MarylandGenerator(WmGenerator):
    """ Generate text using LLaMA and Maryland's watemrarking method. """
    def __init__(self, 
            *args, 
            gam: float = 0.5,
            delta: float = 1.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gam
        self.delta = delta

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
        - add delta to greenlist words' logits
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        logits = self.logits_processor(logits[:, :self.tokenizer.vocab_size], ngram_tokens)#Need to remove the "added_tokens" in recent Phi3 implementations
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1) 
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        vocab_size = self.tokenizer.vocab_size
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            bias = torch.zeros(vocab_size).to(logits.device) # n
            bias[greenlist] = self.delta
            bias = bias.roll(-self.payload)
            logits[ii] += bias # add bias to greenlist words
        return logits


"""
WaterMax variants
"""
class NewGaussianSentenceWm(WmDetector):
    def __init__(self, 
            tokenizer,
            split_len: int,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.rng = npr.default_rng(seed)
        if self.ngram ==0:
            self.change_rng_seed(self.seed)
            self.fixed_u = self.rng.standard_normal(len(tokenizer))
            self.fixed_u[tokenizer.eos_token_id] = np.nan
            self.fixed_u[tokenizer.unk_token_id] = np.nan
        else: self.fixed_u = None
        self.split_len = split_len
        self.hash_size=2**64 -1
        self.ngram = ngram
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = int(np.mod(seed * self.salt_key + i, self.hash_size))
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed
    def change_rng_seed(self, seed): #BEWARE: Impure function based on side-effect
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state
    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of token
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x,add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            #print(total_len)
            start_pos = 0
            rts = []
            seen_ntuples = set()
            if self.ngram > 0:
                n_splits = total_len//self.split_len
                for cur_split in range(n_splits+1):
                    
                    cur_split_tokens_id = tokens_id[ii][self.ngram-1:][self.split_len*cur_split:self.split_len*(cur_split+1)]
                    #print(self.tokenizer.decode(cur_split_tokens_id))
                    cur_split_size = np.min([self.split_len, len(cur_split_tokens_id)])
                    if cur_split_size >0:
                        rt = [] #np.zeros(cur_split_size)
                        #start_pos = self.split_len*cur_split
                        for cur_pos in range(start_pos, cur_split_size):
                            if (cur_pos < self.ngram-1):
                                if (cur_split ==0):
                                    ngram_tokens = tokens_id[ii][:self.ngram-1][-self.ngram+cur_pos+1:] + cur_split_tokens_id[:cur_pos+1] 
                                else:
                                    prev_split_tokens_id = tokens_id[ii][self.ngram-1:][self.split_len*(cur_split-1):self.split_len*cur_split]
                                    ngram_tokens = prev_split_tokens_id[-self.ngram+cur_pos+1:] + cur_split_tokens_id[:cur_pos+1]

                            else: ngram_tokens = cur_split_tokens_id[cur_pos-self.ngram+1:cur_pos+1]
                            #print(ngram_tokens)
                            #print(tokenizer.batch_decode([ngram_tokens]))
                            if scoring_method == 'none':
                                 rt.append(self.score_tok(ngram_tokens))
                            elif scoring_method == 'v1':
                                #tup_for_unique = self.get_seed_rng(ngram_tokens)
                                tup_for_unique = tuple(ngram_tokens)
                                if not tup_for_unique in seen_ntuples:
                                    seen_ntuples.add(tup_for_unique)
                                    rt.append(self.score_tok(ngram_tokens))
                            #rt = rt.numpy()[:payload_max+1]
                        rts.append([np.nansum(rt)/np.sqrt(len(rt))])

                    

            elif self.ngram == 0:
                n_splits = total_len//self.split_len
                for cur_split in range(n_splits+1):
                    ngram_tokens = tokens_id[ii][self.split_len*cur_split:self.split_len*(cur_split+1)]
                    if len(ngram_tokens) > 0:
                        if scoring_method == 'none':
                            scores = [self.score_tok_ngram0(ngram_tokens)/np.sqrt(len(ngram_tokens))]
                        elif scoring_method == 'v1':
                            scores = []
                            for t in ngram_tokens:
                                tup_for_unique = tuple([t])
                                if not tup_for_unique in seen_ntuples:
                                    seen_ntuples.add(tup_for_unique)
                                    scores.append(self.fixed_u[t])
                            scores = [np.nansum(scores)/np.sqrt(len(scores))]
                                
                        rts.append(scores)
            elif self.ngram == -1:
                
                n_splits = total_len//self.split_len
                for cur_split in range(n_splits+1):
                    ngram_tokens = tokens_id[ii][self.split_len*cur_split:self.split_len*(cur_split+1)]
                    if len(ngram_tokens) > 0:
                        scores = [self.score_tok_ngrammax(ngram_tokens)]
                        rts.append(scores)
            else:
                raise Exception("Unimplemented ngram mode")
            #print(rts)
            score_lists.append(rts)
        return score_lists
    
    def score_tok_ngram0(self,ngram_tokens):
        scores = np.nansum(self.fixed_u[ngram_tokens])
        return scores
    def score_tok_ngrammax(self, ngram_tokens):
        """ 

        """
        #self.rng.manual_seed(seed)
        #print(ngram_tokens)
        seed = self.get_seed_rng(ngram_tokens)
        
        u = np.zeros(len(ngram_tokens))
        for i in range(len(ngram_tokens)):
            self.change_rng_seed(seed+i) #SIDE EFFECT: changes rng state, reset to base state of the seed
            u[i] = self.rng.standard_normal()
        #print(ngram_tokens,seed)
        scores = np.nansum(u)/np.sqrt(len(ngram_tokens))
        return scores
    def score_tok(self, ngram_tokens):
        """ 

        """
        seed = self.get_seed_rng(ngram_tokens)
        self.change_rng_seed(seed) #SIDE EFFECT: changes rng state, reset to base state of the seed
        u = self.rng.standard_normal()
        #print(ngram_tokens,seed)
        
        return u
    
    def get_pvalues(self, scores: float,  eps: float = 1e-200):
        """ from cdf of a gamma distribution """
        #if score < 0: pvalue = 1
        #else: pvalue = np.exp(-score**2/(2*ntoks))
        #print(np.sum(~np.isnan(scores)))
        pvalue = gamma.cdf(-np.nansum(norm.logcdf(scores)),a=np.sum(~np.isnan(scores)))
        return max(pvalue, eps)

class NewRobustWmSentenceGenerator():
    
    def __init__(self, 
            model, 
            tokenizer,
            seed :int = 0,
            seeding:str = 'hash',
            salt_key:int = 35317,
            ngram:int = 3,
            num_seq:int=10,
            eos_value = np.inf, # How to score outputs containing a single eos token, np.nan will skip them, while np.inf will stop the generation
                ): 
            
        self.model = model
        self.tokenizer = tokenizer
        self.salt_key = salt_key
        self.seed = seed
        self.rng = npr.default_rng(seed)
        self.ngram = ngram
        self.seeding = seeding
        self.num_seq = num_seq
        self.scores = []
        self.generate_counter =0
        self.hash_size = (2 ** 64 - 1)
        self.eos_value = eos_value
        if self.ngram ==0:
            self.change_rng_seed(self.seed)
            self.fixed_u = self.rng.standard_normal(len(tokenizer))


        else: self.fixed_u = None
        self.seen_ngram = []
    
    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    def change_rng_seed(self, seed): #BEWARE: Impure function based on side-effect
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state
    
    def get_seed_rng(
        self, 
        input_ids
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for jj,i in enumerate(input_ids):
                seed = int(np.mod(seed * self.salt_key + i, self.hash_size))
                #print(jj,i,seed)
        elif self.seeding == 'additive':
            seed = self.salt_key * np.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = np.min(seed)
        return seed
    
    def scoring_outputs(self, input_len, outputs, chaining_ngrams=True):
        scores = np.zeros(len(outputs))
        if self.ngram > 0:
            us_tmp = []
            for ii in range(len(outputs)):
                us_tmp.append([])
                input_offset = input_len[ii//self.num_seq] #len(inputs['input_ids'][ii//self.num_seq])
                if len(outputs[ii][input_offset:]) != 0:
                    Xagg = 0
                    for jj in range(input_offset, len(outputs[ii])): #Offset by prompt from tokenizer to synchronize properly
                        if not chaining_ngrams and (jj < input_offset + self.ngram -1): 
                            ngram_w = outputs[ii][input_offset:][-self.ngram+(jj-input_offset)+1:] + outputs[ii][input_offset:][:(jj-input_offset)+1]
                        else:
                            ngram_w = outputs[ii][jj-self.ngram+1:jj+1]
                        if tuple(ngram_w) in self.seen_ngram[ii]:
                            continue
                        else:
                            #print(ngram_w)
                            seed = self.get_seed_rng(ngram_w)
                            self.change_rng_seed(seed) #SIDE EFFECT: changes rng state, reset to base state of the seed
                            u = self.rng.standard_normal()
                            us_tmp[ii].append(u)
                            Xagg += u # Allows a Gaussian hypo under H0 and Gumbel-type (EVD) under H1

                            self.seen_ngram[ii].add(tuple(ngram_w))
                    if len(us_tmp[ii]) > 0:
                        scores[ii] = Xagg/np.sqrt(len(us_tmp[ii]))
                    

                        
                        
                    else: scores[ii] = -np.inf # No scorable ngrams, reject text
                else:
                    scores[ii] = self.eos_value # Output is empty, meaning it only contained special characters (i.e <eos>)
            #self.us.append(us_tmp)
            
                 
                
        elif self.ngram == 0:
            for ii in range(len(outputs)):
                input_offset = input_len[ii//self.num_seq]
                if len(outputs[ii][input_offset:]) != 0:
                    #us_tmp = self.fixed_u[outputs[ii][input_offset:]]
                    us_tmp = []
                    for jj in range(input_offset, len(outputs[ii])):
                        ngram_w = outputs[ii][jj]
                        if tuple([ngram_w]) in self.seen_ngram[ii]:
                            continue
                        else:
                            us_tmp.append(self.fixed_u[ngram_w])
                            self.seen_ngram[ii].add(tuple([ngram_w]))

                    #print( self.tokenizer.decode(outputs[ii][input_offset:],skip_special_tokens=True))
                    if len(us_tmp) > 0:
                        Xagg = np.sum(us_tmp)/np.sqrt(len(us_tmp)) #Normalizing so that the aggregated score is a centered gaussian
                    else:
                        Xagg = -np.inf

                else:
                    Xagg = self.eos_value
                scores[ii] = Xagg 
        elif self.ngram==-1:
            for ii in range(len(outputs)):
                input_offset = input_len[ii//self.num_seq] #len(inputs['input_ids'][ii//self.num_seq])
                ngram_tokens = outputs[ii][input_offset:]
                if len(ngram_tokens ) != 0: # If we have not already reached <eos>
                    us_tmp = np.zeros(len(ngram_tokens ))
                    seed = self.get_seed_rng(ngram_tokens)
                    #We add more randomness in case of repeating tokens
                    for i in range(len(ngram_tokens)):
                        self.change_rng_seed(seed+i) #SIDE EFFECT: changes rng state, reset to base state of the seed
                        us_tmp[i] = self.rng.standard_normal()

                
                
                    Xagg = np.sum(us_tmp)/np.sqrt(us_tmp.size) #Normalizing so that the aggregated score is a centered gaussian
                else:
                    Xagg = self.eos_value
                scores[ii] = Xagg 
        return(scores)
    
    
    def contains_eos(self, t) -> bool:
        #print(t)
        if self.tokenizer.eos_token_id in t:
            return(True)
        else:
            return(False)
        
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        top_p: float = 0.95,
        do_sample:bool =True,
        num_beams: int =1,
        temperature:float =0.85,
        n_splits:int = 1,
        beam_chunk_size=1
        #num_return_sequences: int = 5
    ) -> List[str]: 
        split_len = max_gen_len//n_splits
        finished_text = np.zeros(len(prompts),dtype=bool)
        res = [[] for _ in range(len(prompts))]
        self.seen_ngram = [set() for _ in range(len(prompts)*self.num_seq)]

        # We start by generating the first 'ngram' tokens since they won't be used by the detector, this will ensure more diversity during the beam-search pahse of chunk generation
        if self.ngram > 0: 
            inputs =self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to(device)
            h_outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=1, num_return_sequences=1,max_new_tokens=self.ngram-1)
            prompts= self.tokenizer.batch_decode(h_outputs,skip_special_tokens=True)


        
        for k in range(n_splits):
            if k == 0:
                inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to(device)
                
            else:
                inputs = self.tokenizer(res, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to(device)
                
            
            input_len = np.array([len(np.array(p)[np.argwhere(np.array(p) != self.tokenizer.eos_token_id)[0][0]:]) for p in inputs['input_ids'].cpu()])
            
            #Ensure a different starting point for each text in order to guarantee inter-text independence
            if beam_chunk_size > 0:
                start_outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=False, num_beams=self.num_seq, num_return_sequences=self.num_seq,max_new_tokens=beam_chunk_size)
                inputs = {'input_ids': start_outputs, 'attention_mask': torch.cat((inputs['attention_mask'], torch.ones(len(prompts),beam_chunk_size).to(device)),1).repeat_interleave(self.num_seq,0)}
                if k < n_splits-1:
                    outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=num_beams, num_return_sequences=1,max_new_tokens=split_len - beam_chunk_size)
                elif split_len - beam_chunk_size - self.ngram  +1> 0:    
                    outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=num_beams, num_return_sequences=1,max_new_tokens=split_len - beam_chunk_size - self.ngram +1)

            else:
                if k < n_splits-1:
                    outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=num_beams, num_return_sequences=self.num_seq,max_new_tokens=split_len)
                elif split_len - beam_chunk_size - self.ngram  +1> 0:
                    outputs = self.model.generate(**inputs, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=num_beams, num_return_sequences=self.num_seq,max_new_tokens=split_len- self.ngram+1)

            is_eos_split = np.array(list(map(lambda x,y : self.contains_eos(x[input_len[y]:]), outputs,np.repeat(np.arange(0,len(prompts)),self.num_seq))))
            
            #outputs_sizes = np.array(list(map(lambda x,y : len(x[input_len[y]:]), outputs,np.repeat(np.arange(0,len(prompts)),self.num_seq))))

            # We need to decode re-encode in order to prevent some double tokens (such as \n\n) to be read as one at the detector side
            # Actually, we can't do much if the tokenizer adds or substract a token when decoding/recoding except reject the text
            # For Llama2, beware of things such as function name e.g factorial(num) will be outputed as a single token factorial( but recoded as two tokens factorial( 
            
            final_outputs= self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
           
            outputs = self.tokenizer(final_outputs,add_special_tokens=False)['input_ids']

            #if k==0: input_len -= 1 # Remove the starting <s> which was striped by skipping special_tokens 
            renc_outputs_sizes = np.array(list(map(lambda x,y : len(x[input_len[y]:]), outputs,np.repeat(np.arange(0,len(prompts)),self.num_seq))))
            

            scores = self.scoring_outputs(input_len, outputs)

            valid_split_mask = (is_eos_split) | (split_len == renc_outputs_sizes) #If the split contains an eos, we can select it as the final split of the text
            if k < n_splits-1: scores[~valid_split_mask] = -np.inf # We reject pathological re-encodings, but not for the last split since it won't desynchronize the decoder
            scores = scores.reshape(len(prompts), self.num_seq, -1)
            is_eos_split = is_eos_split.reshape(len(prompts), self.num_seq)
            idxs = np.nanargmax(scores,axis=1) #Will raise an exception if all NaN !

            for ii,idx in enumerate(idxs):
                #print(idx)
                if not finished_text[ii]:
                    for jj in range(self.num_seq):
                        if jj != idx[0]: self.seen_ngram[ii*self.num_seq + jj] = self.seen_ngram[ii*self.num_seq + idx[0]].copy()
                        
                    res[ii] = final_outputs[ii*self.num_seq:(ii+1)*self.num_seq][idx[0]]
                    if is_eos_split[ii][idx[0]]: 
                        #print(f"Finished text: {ii}")
                        finished_text[ii] = True


        self.generate_counter +=1
        return (res)