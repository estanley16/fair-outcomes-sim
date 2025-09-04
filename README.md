<div align="center">
  
# Connecting algorithmic fairness and fair outcomes in a sociotechnical simulation study of AI-assisted healthcare

</div>

## Abstract
Artificial intelligence (AI) has vast potential for improving healthcare delivery, but concerns over biases in these systems have raised important questions regarding fairness when deployed clinically. Most prior studies on fairness in clinical AI focus solely on performance disparities between subpopulations, which often fall short of connecting the technical outputs of AI systems with sociotechnical outcomes. In this work, we present a simulation-based approach to explore how statistical definitions of algorithmic fairness translate to fairness in long-term outcomes, using AI-assisted breast cancer screening as a case example.  We evaluate four fairness criteria and their impact on mortality rates and socioeconomic disparities, while also considering how radiologists’ reliance on AI and patients’ access to healthcare affect outcomes. Our results highlight how algorithmic fairness does not directly translate into fair and equitable outcomes, underscoring the importance of integrating sociotechnical perspectives in order to gain a holistic understanding of fairness in AI.   

<p align="center">
<img src="fig/fig1.png?raw=true" width="600">
</p>



## Overview 


### Running Experiments
`experiments/` contains python files that can be used to reproduce the results from the paper. 

The files are named after the different analyses performed in the paper:
* `run_baseline.py` runs the simulation with the baseline parameters. 
* `run_ai_performance.py` runs the simulation for different degrees of TPR and FPR performance disparities.
* `run_ai_reliance.py` runs the simulation for different values of the AI reliance parameter (alpha). 
* `run_screening.py`, `run_delay.py`, and `run_treatment.py` run the simulations for different socioeconomic index barriers corresponding to the screening, delay, and treatment stages. 
* `run_ai_performance.py` runs the simulation for different TPR and FPR values of the AI system (from supplementary material). 

Running the simulations requires an argument for the algorithmic fairness scenario to use, <i>e.g.</i>:

`python run_baseline.py --ai_fairness EqOdds`

Simulation variables can be modified in `variables.py`.


### Analyzing Results
`analysis/` contains python notebooks that can be used to analyze the results of the experiments. The `.ipynb` filenames correspond to the `.py` experiment filenames.


### Usage
`environment.yml` can be used to create the conda environment that was used in this work.

Change `results_dir` in the `.py` and `.ipynb` files to desired location for saved results.


## Data 
<!-- Source data used to generate the figures in the manuscript has been deposited [here](). -->

## Citation
coming soon :-) 
