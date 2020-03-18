# MWDMSP
The Code is created based on the method described in the following paper:
Yuan Yuan, Siyuan Wang, Shanshan Wang, Qiegen Liu. Multi-Wavelet guided Deep Mean-Shift Prior for Image Restoration


## Motivation
Image restoration is essentially recognized as an ill-posed problem. A feasible solution is incorporating various priors into restoration 
procedure as constrained conditions. Wavelet transform is a very classical tool associated with regularized inverse problem.

### Figs
![repeat-MWDMSP](https://github.com/yqx7150/HFDAEP/blob/master/figs/forward%20and%20backward.png)
Fig. 1. Flow-chart of the MWDMSP network.

![repeat-MWDMSP](https://github.com/yqx7150/HFDAEP/blob/master/figs/itermri.png)
Fig. 2. Illustration of the MWDMSP network as a priori information for the iterative process IR. 

### Table

![repeat-MWDMSP](https://github.com/yqx7150/HFDAEP/blob/master/figs/table.png)

### Deblured visual Comparisons
![repeat-MWDMSP](https://github.com/yqx7150/HFDAEP/blob/master/figs/result.png)
Fig. 3. Visual quality comparison of image deblurring. Top line: image “Barbara” on Gaussian kernel: 17*17 ,sigma=2.55 , Middle line: image “Peppers” on Gaussian kernel: 17*17 ,sigma=7.65, Bottom line: image “Boats” on Gaussian kernel:25*25, sigma=7.65 . From left to right: noisy and blurred image, the deblurred images obtained by LevinSps, EPLL, DAEP, DMSP, DPE, and MWDMSP.

### CS visual Comparisons
![repeat-MWDMSP](https://github.com/yqx7150/HFDAEP/blob/master/figs/result2.png)
Fig. 4.Recovery results at 15% random sampling (Top) and 10% radial sampling (Bottom). From left to right: Mask pattern, PANO, NLR-CS, BM3DRec, DMSP and MWDMSP.

## Requirements and Dependencies
    caffe
    cuda 8.0
    matconvnet
    
##  deblured
'./DemoMRI/demo_MRI.m' is the demo of HF-DAEP for MRI reconstruction.
## CS
'./DemoCT/demo_CTRec.m' is the demo of HF-DAEP for CT reconstruction.
'./DemoCT/ultilies/generateSystemMatrix.m' is used to generate the system matrix.

The model can be downloaded in 链接：https://pan.baidu.com/s/1Liz4dfCApjPH_uYlUhTvfw 提取码：f5xo 



## Other Related Projects
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)
 
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)
