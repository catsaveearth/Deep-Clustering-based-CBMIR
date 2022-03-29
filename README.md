<h1 align="center">Deep Clustering based CBMIR</h1>
2021.03~2022.06, Senier project, Department of Software, Gachon University <br>
This code performs content based medical image retreival based on deep clustering. Deep clustering refers to SwAV of facebookresearch.
This project was mentored by INFINITT Healthcare Co. Ltd. (http://www.infinitt.com/)
<p align="center">
    <img alt="SwAVpaper" src="https://user-images.githubusercontent.com/50789540/160507759-4d3982b8-872d-4d33-b8a7-fa0580859812.png">
</p>
<br><br>

# Deep clustering Reference
I referenced the following paper and code.
<p align="center">
    <img alt="SwAVpaper" src="https://user-images.githubusercontent.com/50789540/160507637-be5ebbfa-0612-4c2b-80c8-8dd4dfbd7b33.png" width="500px">
</p>
<p align="center">
    <img alt="SwAV" src="https://user-images.githubusercontent.com/50789540/160504930-2a42b521-608a-4fa8-a3d5-e817c2d6b224.png">
</p>
https://github.com/facebookresearch/swav


# Requirements
* Python 3.6
* PyTorch install = 1.4.0
* torchvision
* CUDA 10.1
* Other dependencies: scipy, pandas, numpy
* Apex with CUDA extension

```
// install Apex with CUDA extension
// from https://github.com/facebookresearch/swav/issues/18#issuecomment-748123838

git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext

python -c 'import apex; from apex.parallel import LARC' # should run and return nothing
python -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)' # should run and return apex.parallel.optimized_sync_batchnorm
```

# How to Run
Go to Wiki


# Dataset
Liver segmentation 3D-IRCADb-01. [3d-ircadb-01](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) <br>
<p align="center">
    <img alt="SwAVpaper" src="https://user-images.githubusercontent.com/50789540/160507860-400991f3-07d0-4927-9496-80f1398659d4.png" width="500px">
</p>
The 3D-IRCADb-01 database is composed of the 3D CT-scans of 10 women and 10 men with hepatic tumours in 75% of cases. Provides masks for artery, bone, kidneys, liver, lungs, spleen, etc.

# Related Mobile Dicom Viewer
https://github.com/awholeneworld/MobileDicomViewer

# Presentation Videos
* [Proposal Presentation](https://www.youtube.com/watch?v=21LakDM6ZPU)
* [Implementation Presentation](https://www.youtube.com/watch?v=rVPwWlqNtRU)


# License
### This project: [CC-BY-NC](https://github.com/catsaveearth/Deep-Clustering-based-CBMIR/blob/main/README.md) <br>
![image](https://user-images.githubusercontent.com/50789540/160509321-23bd9d1f-8511-4eca-a3e8-3918264f7fa2.png)
```
Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.
.
.
.
```

### SwAV License : [CC-BY-NC](https://github.com/facebookresearch/swav/edit/main/LICENSE) <br>
![image](https://user-images.githubusercontent.com/50789540/160509321-23bd9d1f-8511-4eca-a3e8-3918264f7fa2.png)

```
Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.
.
.
.
```
