# Awesome-LLM-Self-Improvement
A curated list of awesome LLM Inference-Time Self-Improvement (**ITSI**, pronounced *"itsy"*) papers from our recent survey: 
[*A Survey on Large Language Model Inference-Time Self-Improvement*](https://arxiv.org/pdf/2412.14352).

> 📬 **Contacts:** Questions or comments? Send us an email: Xiangjue Dong (`xj.dong@tamu.edu`) & Maria Teleki (`mariateleki@tamu.edu`).

> ⭐️ **Want to contribute?** Send us a pull request! New papers are welcome.

## Independent Self-Improvement
Independent Self-Improvement is achieving improvements in performance using the model's own frozen parameters without additional training -- i.e., by modifying the decoding process, increasing efficiency, sampling multiple candidate generations, and isolating layers or neurons.

### Constrained Decoding

Constrained decoding guides the generation process via hard constraints or soft constraints.

#### Hard Constraint

* NeuroLogic*: [NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics](https://aclanthology.org/2022.naacl-main.57/) (Lu et al., NAACL 2022) ([Code](https://github.com/GXimingLu/a_star_neurologic))

* NeuroLogic: [NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints](https://aclanthology.org/2021.naacl-main.339/) (Lu et al., NAACL 2021) ([Code](https://github.com/GXimingLu/neurologic_decoding))

* Control-DAG: [Control-DAG: Constrained Decoding for Non-Autoregressive Directed Acyclic T5 using Weighted Finite State Automata](https://aclanthology.org/2024.naacl-short.42) (Chen et al., NAACL 2024) ([Code](https://github.com/EriChen0615/ControlDAG))

#### Soft Constraint

* Penalty Decoding: [Penalty Decoding: Well Suppress the Self-Reinforcement Effect in Open-Ended Text Generation](https://aclanthology.org/2023.emnlp-main.78) (Zhu et al., EMNLP 2023) ([Code](https://github.com/zwhong714/penalty_decoding))

* IPS (Isotropic and Proximal Search): [Fine-grained Conversational Decoding via Isotropic and Proximal Search](https://aclanthology.org/2023.emnlp-main.5) (Yao et al., EMNLP 2023) ([Code](https://github.com/starrYYxuan/IPS))

### Contrastive Decoding

Contrastive decoding adjusts the next-token probability based on differences in logits.

#### Faithfulness and Hallucinations.

* PMI-Decode: [Pointwise Mutual Information Based Metric and Decoding Strategy for Faithful Generation in Document Grounded Dialogs](https://aclanthology.org/2023.emnlp-main.639/) (Nandwani et al., EMNLP 2023) ([Code](https://github.com/ynandwan/pmi-faith))

* LCD: [Mitigating Hallucinations in Large Vision-Language Models (LVLMs) via Language-Contrastive Decoding (LCD)](https://aclanthology.org/2024.findings-acl.359/) (Manevich & Tsarfaty, Findings 2024) ([Code](https://github.com/DAMO-NLP-SG/VCD))

* Anti-LM: [Anti-LM Decoding for Zero-shot In-context Machine Translation](https://aclanthology.org/2024.findings-naacl.216/) (Sia et al., NAACL Findings 2024) ([Code](https://github.com/suzyahyah/icl_Anti-LM_decoding))

* DoLA: [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://openreview.net/forum?id=Th6NyL07na) (Chuang et al., ICLR 2024) ([Code](https://github.com/voidism/DoLa))

#### Repetition, Coherence, and Diversity. 

* Look-Back: [Look-back Decoding for Open-Ended Text Generation](https://aclanthology.org/2023.emnlp-main.66/) (Xu et al., EMNLP 2023) ([Code](https://github.com/xunannancy/LookBackDecoding))

* Adaptive Decoding: [Improving Open-Ended Text Generation via Adaptive Decoding](https://openreview.net/forum?id=aXD94eATtT) (Zhu et al., ICML 2024) ([Code](https://github.com/zwhong714/adaptive_decoding))

### Minimum Bayes-Risk Decoding

Unlike MAP decoding, Minimum Bayes-Risk (MBR) decoding selects the output sentence that maximizes the *expected utility* over the set of translation hypotheses (Kumar and Byrne, 2004).

#### Clustering

* DMBR: [Generating Diverse and High-Quality Texts by Minimum Bayes Risk Decoding](https://aclanthology.org/2024.findings-acl.503/) (Jinnai et al., ACL Findings 2024) ([Code](https://github.com/CyberAgentAILab/diverse-mbr/))

* CBMBR: [Centroid-Based Efficient Minimum Bayes Risk Decoding](https://aclanthology.org/2024.findings-acl.654/) (Deguchi et al., ACL Findings 2024) ([Code](https://github.com/naist-nlp/mbrs))

#### Matrix Approximation

* PMBR: [Efficient Minimum Bayes Risk Decoding using Low-Rank Matrix Completion Algorithms](https://openreview.net/forum?id=8iPobEKUUA) (Trabelsi et al., NeurIPS 2024)

#### Other

* Pruning MBR: [Faster Minimum Bayes Risk Decoding with Confidence-based Pruning](https://aclanthology.org/2023.emnlp-main.767/) (Cheng & Vlachos, EMNLP 2023) ([Code](https://github.com/juliusc/pruning_mbr))

* AMBR: [Hyperparameter-Free Approach for Faster Minimum Bayes Risk Decoding](https://aclanthology.org/2024.findings-acl.505/) (Jinnai & Ariu, ACL Findings 2024) ([Code](https://github.com/CyberAgentAILab/adaptive-mbr))

* MBMBR: [Model-Based Minimum Bayes Risk Decoding for Text Generation](https://openreview.net/forum?id=qDUaH9xHVV) (Jinnai et al., ICML 2024) ([Code](https://github.com/CyberAgentAILab/model-based-mbr))

### Parallel Decoding

Parallel decoding generates multiple tokens simultaneously during the decoding phases for faster generation, rather than sequentially.

* HGJ: [Accelerating Transformer Inference for Translation via Parallel Decoding](https://aclanthology.org/2023.acl-long.689/) (Santilli et al., ACL 2023) ([Code](https://github.com/teelinsan/parallel-decoding))

* Arithmetic Sampling: [Arithmetic Sampling: Parallel Diverse Decoding for Large Language Models](https://proceedings.mlr.press/v202/vilnis23a.html) (Vilnis et al., ICML 2023) ([Code](https://github.com/google-research/google-research/tree/master/arithmetic_sampling))

* Lookahead Decoding: [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://openreview.net/forum?id=eDjvSFOkXw) (Fu et al., ICML 2024) ([Code](https://github.com/hao-ai-lab/LookaheadDecoding))

* SoT: [Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://openreview.net/forum?id=mqVgBbNCm9) (Ning et al., ICLR 2024) ([Code](https://github.com/imagination-research/sot))

### Sampling-Based Decoding

Sampling-based methods introduce randomness for token selection to generate diverse text or sample multiple generation paths from the model.

#### Open-Ended Generation.

* BAT: [Closing the Curious Case of Neural Text Degeneration](https://openreview.net/forum?id=dONpC9GL1o) (Finlayson et al., ICLR 2024) ([Code](https://github.com/mattf1n/basis-aware-threshold))

* DAEMON: [Language Model Decoding as Direct Metrics Optimization](https://openreview.net/forum?id=488A64eOf6) (Ji et al., ICLR 2024)

#### Reasoning.

* Self-Consistency: [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/forum?id=1PL1NIMMrw) (Wang et al., ICLR 2023)

* ESC: [Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](https://openreview.net/forum?id=ndR8Ytrzhh) (Li et al., ICLR 2024) ([Code](https://github.com/Yiwei98/ESC))

#### Other.

* ASAp: [Grammar-Aligned Decoding](https://openreview.net/forum?id=5G7ve8E1Lu) (Park et al., NeurIPS 2024) ([Code](https://github.com/ebmoon/transformers-GAD))


### Tree-Search-based Decoding

Planning algorithms, such as Monte-Carlo tree search (MCTS), have also been applied to identify optimal text outputs for various tasks.

* PG-TD: [Planning with Large Language Models for Code Generation](https://openreview.net/forum?id=Lr8cOOtYbfL) (Zhang et al., ICLR 2023) ([Code](https://github.com/shunzh/Code-AI-Tree-Search))

* GDP-Zero: [Prompt-Based Monte-Carlo Tree Search for Goal-oriented Dialogue Policy Planning](https://aclanthology.org/2023.emnlp-main.439/) (Yu et al., EMNLP 2023) ([Code](https://github.com/jasonyux/GDPZero))

* RAP: [Reasoning with Language Model is Planning with World Model](https://aclanthology.org/2023.emnlp-main.507/) (Hao et al., EMNLP 2023) ([Code](https://github.com/maitrix-org/llm-reasoners))

### Model-level Decoding

Model-level methods operate inside the intermediate layers of the model.

* ACD: [The Benefits of Bad Advice: Autocontrastive Decoding across Model Layers](https://aclanthology.org/2023.acl-long.580/) (Gera et al., ACL 2023) ([Code](https://github.com/IBM/auto-contrastive-generation))

* Self-Speculative Decoding: [Draft& Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://aclanthology.org/2024.acl-long.607/) (Zhang et al., ACL 2024) ([Code](https://github.com/dilab-zju/self-speculative-decoding))

* SLED: [SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models](https://openreview.net/forum?id=t7wvJstsiV) (Zhang et al., NeurIPS 2024)

* Language-Specific Neurons: [On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons](https://aclanthology.org/2024.naacl-long.384/) (Kojima et al., NAACL 2024) ([Code](https://github.com/kojima-takeshi188/lang_neuron))

* Overthinking and False Induction Heads: [Overthinking the Truth: Understanding how Language Models Process False Demonstrations](https://openreview.net/forum?id=Tigr1kMDZy) (Halawi et al., ICLR 2024) ([Code](https://github.com/dannyallover/overthinking_the_truth))

## Context-Aware Self-Improvement
Context-Aware Self-Improvement enhances performance using specialized prompt-based or retrieval-based techniques. 

### Prompting
Prompting uses carefully crafted prompts to enable few-shot or zero-shot learning without parameter updates (Liu et al., 2023).

#### Reasoning.

* CoT Prompting: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://openreview.net/forum?id=_VjQlMeSB_J) (Wei et al., NeurIPS 2022)

* Zero-Shot CoT Prompting: [Large Language Models are Zero-Shot Reasoners](https://openreview.net/forum?id=e2TBb5y0yFf) (Kojima et al., NeurIPS 2024) ([Code](https://github.com/kojima-takeshi188/zero_shot_cot))

* EchoPrompt: [EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning](https://aclanthology.org/2024.naacl-short.35/) (Mekala et al., NAACL 2024) ([Code](https://github.com/rajasekharmekala/echoprompt))

#### Others.

* DecodingTrust: [DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models](https://openreview.net/forum?id=kaHpo8OZw2) (Wang et al., NeurIPS 2023) ([Code](https://github.com/AI-secure/DecodingTrust))

* Generation Exploitation Attack: [Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation](https://openreview.net/forum?id=r42tSSCHPh) (Huang et al., ICLR 2024) ([Code](https://github.com/Princeton-SysML/Jailbreak_LLM))

* URIAL: [The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning](https://openreview.net/forum?id=wxJ0eXwwda) (Lin et al., ICLR 2024) ([Code](https://github.com/Re-Align/urial))


### Disturbed Prompt

Disturbed prompt methods use a specialized prompt (e.g., disturbance or noisy instruction) to obtain and contrast the model's probability distributions *with and without the specialized prompt* during the decoding.

* ICD: [Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding](https://aclanthology.org/2024.findings-acl.937/) (Wang et al., ACL Findings 2024)

* CAD: [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://aclanthology.org/2024.naacl-short.69/) (Shi et al., NAACL 2024) ([Code](https://github.com/xhan77/context-aware-decoding))

* ID: [Instructive Decoding: Instruction-Tuned Large Language Models are Self-Refiner from Noisy Instructions](https://openreview.net/forum?id=LebzzClHYw) (Kim et al., ICLR 2024)

* COIECD: [Discerning and Resolving Knowledge Conflicts through Adaptive Decoding with Contextual Information-Entropy Constraint](https://aclanthology.org/2024.findings-acl.234/) (Yuan et al., ACL Findings 2024) ([Code](https://github.com/Stacy027/COIECD))

* ROSE: [ROSE Doesn’t Do That: Boosting the Safety of Instruction-Tuned Large Language Models with Reverse Prompt Contrastive Decoding](https://aclanthology.org/2024.findings-acl.814/) (Zhong et al., ACL Findings 2024)

### Retrieval-Based
Retrieval-based methods obtain information from existing corpora or construct a retrieval datastore.

* *k*NN-LM: [Generalization through Memorization: Nearest Neighbor Language Models](https://openreview.net/forum?id=HklBjCEKvH) (Khandelwal et al., ICLR 2020) ([Code](https://github.com/urvashik/knnlm))

* NEST: [Nearest Neighbor Speculative Decoding for LLM Generation and Attribution](https://openreview.net/forum?id=Ni9kebsSTt) (Li et al., NeurIPS 2024)

* REST: [REST: Retrieval-Based Speculative Decoding](https://aclanthology.org/2024.naacl-long.88/) (He et al., NAACL 2024) ([Code](https://github.com/FasterDecoding/REST))

* RTD: [Reference Trustable Decoding: A Training-Free Augmentation Paradigm for Large Language Models](https://openreview.net/forum?id=QHRLFdhkLu) (Luohe et al., NeurIPS 2024)

* Multi-Input CD: [Enhancing Contextual Understanding in Large Language Models through Contrastive Decoding](https://aclanthology.org/2024.naacl-long.237/) (Zhao et al., NAACL 2024) ([Code](https://github.com/amazon-science/ContextualUnderstanding-ContrastiveDecoding))

## Model-Aided Self-Improvement
Model-Aided Self-Improvement enhances performance with an external (often small) model.

### Expert and/or Anti-Expert
Expert and/or Anti-Expert models -- specialized or not in a particular task -- provide logits or probability distributions, which are then contrasted or incorporated during decoding, or for otherwise guiding the decoding process via scoring.

#### Toxicity
* DExperts: [DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts](https://aclanthology.org/2021.acl-long.522) (Liu et al., ACL-IJCNLP 2021) ([Code](https://github.com/alisawuffles/DExperts))

* MIL-Decoding: [MIL-Decoding: Detoxifying Language Models at Token-Level via Multiple Instance Learning](https://aclanthology.org/2023.acl-long.11) (Zhang & Wan, ACL 2023) ([Code](https://github.com/pkulcwmzx/Detoxification))

#### Machine Translation
* PSGD: [Easy Guided Decoding in Providing Suggestions for Interactive Machine Translation](https://aclanthology.org/2023.acl-long.434) (Wang et al., ACL 2023) ([Code](https://github.com/ZhenYangIACAS/WeTS))

* CoDec: [Improving Machine Translation with Large Language Models: A Preliminary Study with Cooperative Decoding](https://aclanthology.org/2024.findings-acl.786) (Zeng et al., Findings 2024) ([Code](https://github.com/lemon0830/CoDec))

* LiBS: [Language-Informed Beam Search Decoding for Multilingual Machine Translation](https://aclanthology.org/2024.findings-acl.932) (Yang et al., Findings 2024) ([Code](https://github.com/yilinyang7/fairseq_multi_fix))

* CODEC: [Constrained Decoding for Cross-lingual Label Projection](https://openreview.net/forum?id=DayPQKXaQk) (Le et al., ICLR 2024) ([Code](https://github.com/duonglm38/Codec))

#### Alignment
* MOD: [Decoding-Time Language Model Alignment with Multiple Objectives](https://openreview.net/forum?id=3csuL7TVpV) (Shi et al., NeurIPS 2024) ([Code](https://github.com/srzer/MOD))

* Transfer $Q^*$: [Transfer Q-star : Principled Decoding for LLM Alignment](https://openreview.net/forum?id=5PrShrKxoX) (Chakraborty et al., NeurIPS 2024) ([Code](https://github.com/umd-huang-lab/Transfer-Q))

#### Others
* SafeDecoding: [SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding](https://aclanthology.org/2024.acl-long.303) (Xu et al., ACL 2024) ([Code](https://github.com/uw-nsl/SafeDecoding))

* Superposed Decoding: [Superposed Decoding: Multiple Generations from a Single Autoregressive Inference Pass](https://openreview.net/forum?id=KSOkkHm9I7) (Shen et al., NeurIPS 2024) ([Code](https://github.com/RAIVNLab/SuperposedDecoding))

* GD: [Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents](https://openreview.net/forum?id=JCCi58IUsh) (Huang et al., NeurIPS 2023) ([Code](https://grounded-decoding.github.io/))

* Equilibrium-Ranking: [The Consensus Game: Language Model Generation via Equilibrium Search](https://openreview.net/forum?id=n9xeGcI4Yg) (Jacob et al., ICLR 2024)

### Draft Model
Draft model is used in Speculative Decoding, a method which accelerates LLM inference.

* Speculative Decoding: [Fast Inference from Transformers via Speculative Decoding](https://openreview.net/forum?id=C9NEblP8vS) (Leviathan et al., ICML 2023)

* SpecTr: [SpecTr: Fast Speculative Decoding via Optimal Transport](https://openreview.net/forum?id=SdYHLTCC5J) (Sun et al., NeurIPS 2023)

* GSD: [Graph-Structured Speculative Decoding](https://aclanthology.org/2024.findings-acl.677) (Gong et al., Findings 2024) ([Code](https://github.com/gzhch/gsd))

* SpecExec: [SpecExec: Massively Parallel Speculative Decoding For Interactive LLM Inference on Consumer Devices](https://openreview.net/forum?id=JAhNsZ9dvG) (Svirschevski et al., NeurIPS 2024) ([Code](https://github.com/yandex-research/specexec))

* SEQUOIA: [Sequoia: Scalable and Robust Speculative Decoding](https://openreview.net/forum?id=rk2L9YGDi2) (Chen et al., NeurIPS 2024) ([Code](https://github.com/Infini-AI-Lab/Sequoia))

* SCD: [Speculative Contrastive Decoding](https://aclanthology.org/2024.acl-short.5) (Yuan et al., ACL 2024)

* Theoretical Analysis: [A Theoretical Perspective for Speculative Decoding Algorithm](https://openreview.net/forum?id=wSqpNeMVLU) (Yin et al., NeurIPS 2024)

* Online SD: [Online Speculative Decoding](https://openreview.net/forum?id=BPQHXwVNvl) (Liu et al., ICML 2024) ([Code](https://github.com/LiuXiaoxuanPKU/OSD))

* GliDe: [GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding](https://openreview.net/forum?id=mk8oRhox2l) (Du et al., ICML 2024) ([Code](https://github.com/NonvolatileMemory/GliDe_with_a_CaPE_ICML_24))

### Small LMs/Amateur LMs
In addition to the Draft Model used in Speculative Decoding, other Small LMs -- also referred to as Amateur LMs --  are used to guide LLM generation.

#### Classification Model
* NeuroLogic-A* (P): [Self-Ensemble of N-best Generation Hypotheses by Lexically Constrained Decoding](https://aclanthology.org/2023.emnlp-main.905) (Miyano et al., EMNLP 2023)

* EnDec: [Jailbreak Open-Sourced Large Language Models via Enforced Decoding](https://aclanthology.org/2024.acl-long.299) (Zhang et al., ACL 2024)

* Critic-Driven Decoding: [Critic-Driven Decoding for Mitigating Hallucinations in Data-to-text Generation](https://aclanthology.org/2023.emnlp-main.172) (Lango & Dusek, EMNLP 2023) ([Code](https://github.com/langus0/critic-aware-decoding))

* KCTS: [KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection](https://aclanthology.org/2023.emnlp-main.867) (Choi et al., EMNLP 2023) ([Code](https://github.com/HKUST-KnowComp/Knowledge-Constrained-Decoding))

#### Generative Model
* CD: [Contrastive Decoding: Open-ended Text Generation as Optimization](https://aclanthology.org/2023.acl-long.687) (Li et al., ACL 2023) ([Code](https://github.com/XiangLi1999/ContrastiveDecoding))

* BiLD: [Speculative Decoding with Big Little Decoder](https://openreview.net/forum?id=EfMyf9MC3t) (Kim et al., NeurIPS 2023) ([Code](https://github.com/kssteven418/BigLittleDecoder))

* JAMDEC: [JAMDEC: Unsupervised Authorship Obfuscation using Constrained Decoding over Small Language Models](https://aclanthology.org/2024.naacl-long.87) (Fisher et al., NAACL 2024) ([Code](https://github.com/jfisher52/JAMDecoding))

### Reward Model
The reward model is a fine-tuned LM that evaluates generated responses and assigns scores to guide the decoding process.

* RAD: [Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model](https://aclanthology.org/2023.emnlp-main.721) (Deng & Raffel, EMNLP 2023)

* ARGS: [ARGS: Alignment as Reward-Guided Search](https://openreview.net/forum?id=shgx0eqdw6) (Khanov et al., ICLR 2024) ([Code](https://github.com/deeplearning-wisc/args))

* TS-LLM: [AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training](https://openreview.net/forum?id=C4OpREezgj) (Wan et al., ICML 2024) ([Code](https://github.com/waterhorse1/LLM_Tree_Search))

* Controlled Decoding: [Controlled Decoding from Language Models](https://openreview.net/forum?id=bVIcZb7Qa0) (Mudgal et al., ICML 2024)

### Tool Use/APIs
Interaction with external models also includes tool use -- such as parsers, static analysis tools, and API calls.

* GCD: [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://aclanthology.org/2023.emnlp-main.674) (Geng et al., EMNLP 2023) ([Code](https://github.com/epfl-dlab/GCD))

* NeuroStructural Decoding: [NEUROSTRUCTURAL DECODING: Neural Text Generation with Structural Constraints](https://aclanthology.org/2023.acl-long.528) (Bastan et al., ACL 2023) ([Code](https://stonybrooknlp.github.io/NeuroStructuralDecoding/))

* MGD: [Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context](https://openreview.net/forum?id=qPUbKxKvXq) (Agrawal et al., NeurIPS 2023) ([Code](https://github.com/microsoft/monitors4codegen))

* FLAP: [FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs](https://aclanthology.org/2024.naacl-long.29) (Roy et al., NAACL 2024)

* FANTASE: [FANTAstic SEquences and Where to Find Them: Faithful and Efficient API Call Generation through State-tracked Constrained Decoding and Reranking](https://aclanthology.org/2024.findings-emnlp.359) (Wang et al., Findings 2024) ([Code] (https://github.com/Edillower/FANTASE))

## Other Relevant Survey Paper You May Be Interested In
* [When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs](https://aclanthology.org/2024.tacl-1.78.pdf) (Kamoi et al., TACL 2024) ([code](https://github.com/ryokamoi/llm-self-correction-papers))

* [Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies](https://arxiv.org/pdf/2308.03188) ([code](https://github.com/teacherpeterpan/self-correction-llm-papers))


# Citation
If you find our survey helpful, cite us!
```
@misc{dong2024surveyllminferencetimeselfimprovement,
      title={A Survey on LLM Inference-Time Self-Improvement}, 
      author={Xiangjue Dong and Maria Teleki and James Caverlee},
      year={2024},
      eprint={2412.14352},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14352}, 
}
```
