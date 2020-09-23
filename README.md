# Group Factor Analysis (GFA)

Python implementation of GFA that can be used to uncover multivariate relationships between multiple data sources (e.g. brain and behaviour) in complete and incomplete data sets.

## Description of the files:
- [analysis_syntdata.py](analysis_syntdata.py): run experiments on synthetic data generated with two groups (i.e. two data sources). This module was used to run the experiments on synthetic data described in [1](insert link). 
- [visualization_syntdata.py](visualization_syntdata.py): plot and save the results of the experiments on synthetic data.
This module was used to run the experiments on synthetic data described in [1](insert link).
- [analysis_HCP.py](analysis_HCP.py): this module was used to run the experiments on the HCP data described in [1](insert link). 
- [visualization_HCP.py](visualization_HCP.py): this module was used to plot and save the results the experiments on the HCP data described in [1](insert link). 
- [GFA_original.py](models/GFA_original.py): implementation of the original GFA model proposed in [2](http://proceedings.mlr.press/v22/virtanen12.html)[3](https://www.jmlr.org/papers/v14/klami13a.html).
- [GFA_missingdata.py](models/GFA_missingdata.py): implementation of our GFA extensions proposed in [1](insert link) to handle missing data.
- [utils.py](utils.py): GFA tools for multi-output prediction and missig data prediction.

## Citation
If you want to use this repository for running your experiments, please cite:
- (add citation)

## References
[1](insert link) (add citation)
[2](http://proceedings.mlr.press/v22/virtanen12.html) Seppo Virtanen, Arto Klami, Suleiman Khan, Samuel Kaski. Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, PMLR 22:1269-1277, 2012.
[3](https://www.jmlr.org/papers/v14/klami13a.html) Klami A, Virtanen S, Kaski S. Bayesian Canonical Correlation Analysis. J Mach Learn Res 14:965–1003, 2013.

