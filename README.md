## A practical, effective calculation of gamma difference distributions <br> with open data science tools
*a research paper and supplementary materials - software, notebooks*

by **Martina Hančová, Andrej Gajdoš, Jozef Hanč**  
<martina.hancova@upjs.sk>

### Abstract of the paper

At present, there is still no officially accepted and extensively verified implementation of computing the gamma
difference distribution allowing unequal shape parameters. We explore four computational ways of the gamma difference distribution
with the different shape parameters resulting from time series kriging, a forecasting approach based on the best linear unbiased prediction, and linear mixed models.

The results of our numerical study, with emphasis on using open data science tools, demonstrate that our open tool implemented in high-performance Python(with Numba)
is exponentially fast, highly accurate, and very reliable. It combines numerical inversion of the characteristic function and the trapezoidal rule with the double exponential oscillatory transformation (DE quadrature). At the double 53-bit precision, our tool outperformed the speed of the analytical 
computation based on Tricomi's $U(a, b, z)$ function in CAS software (commercial Mathematica, open SageMath) by 1.5-2 orders. At the default precision of scientific numerical computational tools, it exceeded open SciPy, NumPy, and commercial MATLAB 5-10 times. 

The potential future application of our tool for a mixture of characteristic functions could open new possibilities for fast data analysis based on exact probability distributions in areas like multidimensional statistics, measurement uncertainty analysis in metrology as well as in financial mathematics and risk analysis. 

## Research paper 

This is a pre-print of an article submitted to [Journal of Statistical Computation and Simulation](https://www.tandfonline.com/loi/gscs20). The preprint is available at <https://arxiv.org/abs/2105.04427>.

## Software [![render in nbviewer](misc/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/fdslrm/EBLUP-NE/blob/master/index.ipynb) 

The notebooks folders contain our open codes, Jupyter and Mathematica notebooks from the entire numerical study which are detailed records of our computing 
with explaining narratives ilustrating explored concepts and methods. 

Notebooks can be studied and **viewed** statically in [Jupyter nbviewer](https://nbviewer.jupyter.org/github/fdslrm/EBLUP-NE/blob/master/index.ipynb) [![render in nbviewer](misc/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/fdslrm/EBLUP-NE/blob/master/index.ipynb) with full visualisation. If there is a need, they can be also viewed directly on Github [`index.ipynb`](index.ipynb), also as a raw code. 

For interactive **executing** Jupyter notebooks as live documents without any need to install or compile the software use [CoCalc](https://cocalc.com/) providing interactive computing with our Jupyter notebooks.
 
All source code is distributed under [the MIT license](https://choosealicense.com/licenses/mit/).

## Acknowledgements

This work was supported by the Slovak Research and Development Agency under the contract no. APVV-17-0568 and the Internal Research Grant System of Faculty of Science, P. J. Šafárik University in Košice - project vvgs-pf-2020-1423.
