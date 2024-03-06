<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<div>

<h1>🌊 WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off</h1>

</div>




<!-- ABOUT THE PROJECT -->
## About The Project

This repository is the official implementation of the [WaterMax algorithm](), a LLM watermarking scheme which allows to reach high detectability at close to no cost to text quality and robustness. The repository contains both the generator and detector. It also provides the implementation of other SOTA watermarkings schemes, using the implementation of  [Three Bricks to Consolidate Watermarks for LLMs](https://github.com/facebookresearch/three_bricks).

The code allows the replication of the results of the original WaterMax publication.

Finally, we provide helper functions to make benchmarking compatible with the [Mark My Words](https://github.com/wagner-group/MarkMyWords) benchmark.

<p align="right">(<a href="#readme-top">back to top</a>)</p>






<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

Install the required packages using pip:

  ```sh
  pip install -r requirements.txt
  ```

The main results of this paper use the [HuggingFace weights of LLama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) which require accepting the Meta License.
In order to facilitate the use of the model, we advise downloading the model locally:

```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Make sure your SSH key is properly setup in your user settings.

# https://huggingface.co/settings/keys

git clone git@hf.co:meta-llama/Llama-2-7b-chat-hf
```

Alternatively, the default model used by the scripts is [**mistralai/Mistral-7B-Instruct-v0.2**](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

By default, *bfloat16* tensors are used, this might not be compatible with some GPUs or sometimes leading to slow inference. In this case, use the **--fp16** or **--fp32** arguments to use standard floating-point tensor types. Similarly, quantized weights can be used by adding the **--quantize** argument.

<!-- USAGE EXAMPLES -->
## Usage

### Generating watermarked texts
From the root of the repository:

```sh
python watermax.py  --generate --detect --seed [Seed of the PRNG] --ngram [Hash window size] --n [Number of drafts per chunks] --N [Number of chunks] --gen_len [Max size of generated text] --fp16 --prompts [prompts | path to text file ending in .txt]
```

This will generate a watermark text using WaterMax and run it through the base detector. Both a path to a *.txt* file containg one prompt per line or a list of prompts can be used:

```sh
python watermax.py  --generate --detect --seed=815 --ngram=4 --n=2 --N=2 --fp16  --prompts data/test_prompts.txt
```

```sh
python watermax.py  --generate --detect --seed=815 --ngram=4 --n=2 --N=2 --fp16  --prompts "What was Spinoza's relationship with Leibniz?" "Which philospher spoke about the multicolored cow?"

```

By default, **mistralai/Mistral-7B-Instruct-v0.2** is used. You can specify another model by using the **--model_name** argument:

```sh
python watermax.py --model_name [PATH_TO_MODEL] --generate --detect --seed=815 --ngram=4 --n=2 --N=2 --prompts data/test_prompts.txt
```

The argument accepts both local paths and HuggingFace identifiers.

### Benchmarking WaterMax
From the root of the repository:

``` sh
python test_sentence_wm.py --mode sentence-wm --model_name [PATH_TO_LLAMA2]/Llama-2-7b-chat-hf --generate --detect --seed 815 --ngram 4 --param1 [number of drafts per chunks] --param2 [number of chunks] --batch_size 1 --benches story_reports fake_news invented_stories c4                  

```

Results from the main text can be replicated using the seed 815, results from the appendices use the seed 1015.

#### Possible operations
The *test_sentence_wm.py* script allows to perform 6 different operations:

- **generate**: Performs the watermarking operation specified by **mode** with parameters **param1** and **param2** for prompts defined by **benches**.Outputs a *json* file containing the generated texts.
- **detect**: Performs watermark detection using the method specified by **mode** with parameters **param1** and **param2** for results of **benches**. Outputs a *json* file containing the $p$-value of each text.
- **rate**: Use the model specified by **model_name** to rate the resulting texts of **benches**
- **standardize** (MMW only): Standardize the results and score files to be compatible with the *Mark My Words* benchmark
- **detect_std** (MMW only): Performs detection on texts generated using *Mark My Words*, usually used to compute the p-value of texts attacked using the MMW attack suite
- **standardize_final** (MMW only): Same as standardize but expects a file containing the detection results from **detect_std**.

#### Other quality metrics

Besides LLM ratings, perplexity and MAUVE scores can be computed using:

```sh
python compute_ppl.py --wm [watermark algorithm] --seed 1015 --ngram 4 --gen_len [Max size of generated text] --generator_name Llama-2-7b-chat-hf  --param1 [param1] --param2 [param2] --benches fake_news  story_reports invented_stories
```

```sh
python compute_mauve.py --wm [watermark algorithm] --seed 1015 --ngram 4 --gen_len [Max size of generated text] --model_name Llama-2-7b-chat-hf  --param1 [param1] --param2 [param2] --benches fake_news story_reports invented_stories
```

Beware that MAUVE can only be computed for a given benchmark if non-watermarked texts have already been generated --- see "No watermark" in [Implemented algorithms](#implemented-algorithms).

#### Reusing and extending
The WaterMax generator and detector classes can be found in *models/wm.py*:
- **RobustWmSentenceGenerator**: WaterMax generator
- **GaussianSentenceWm**: WaterMax base detector (detection chunk by chunk)
- **SecureGaussianSentenceWm**: WaterMax robust detector (detection token by token) 

Other prompts/benchmarks can be added by modifying the function *generate_prompts* in *misc/helpers.py*

See *config_model* in *misc/helpers.py* to change how a model is loaded. **Note that a prompt template should be defined** in the function *standardize* from the same file if a new model is added. 


## Implemented algorithms
Other algorithms can used by changing the **mode**, **param1** and **param2** arguments. Note that WaterMax uses the HuggingFace API whereas the other schemes are based on Meta's Llama implementation, hence the different parameters for generating texts without watermarks.
 | Watermark | --mode | --param1 | --param2 | 
 ---|---|---|---|
 |[WaterMax]() | sentence-wm | $n$: Number of drafts per chunk | $N$: Number of chunks |
 |[Kirchenbauer et al.](https://arxiv.org/abs/2301.10226)| kirch | $\delta$:  bias to the logits | $\gamma$: ratio of green-list tokens |
 |[Aaronson et al.](https://scottaaronson.blog/?m=202302)| aaronson | $\theta$: temperature |  - |
 |No watermark (WaterMax)| sentence-wm | 1 |  1  |
 |No watermark (Other schemes) | nowm | - | - |

<p align="right">(<a href="#readme-top">back to top</a>)</p>








<!-- LICENSE -->
## License

Distributed under a CC-BY-NC license. See [LICENSE](LICENSE.txt) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Experiments presented in this paper were carried out using the Grid'5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS,  RENATER and several Universities as well as other organizations (see https://www.grid5000.fr). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Citation
If you find this repository useful, please consider giving a star ⭐ and please cite as:

```
@article{giboulotwatermax2024,
  title={WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off},
  author={Giboulot, Eva and Furon, Teddy},
  journal={},
  year={2024}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
