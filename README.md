## [CVPR 2022] Motion-from-Blur: 3D Shape and Motion Estimation of Motion-blurred Objects in Videos
### [YouTube](https://youtu.be/lRUuLNZx2bY) | [arXiv](https://arxiv.org/abs/2111.14465)

### Prerequisites 

Kaolin is available here: https://github.com/NVIDIAGameWorks/kaolin
Pre-trained DeFMO is implemented directly in kornia.feature.DeFMO

### Running

The code can be easily run by 'python run.py --input video.avi', e.g. check run.sh script.
The results will be written to the output folder.

To run on the fast moving object deblurring benchmark, we used: https://github.com/rozumden/fmo-deblurring-benchmark
For synthetic dataset generation, we used the publicly available implementation from DeFMO authors: https://github.com/rozumden/DeFMO/tree/master/renderer


Reference
------------
If you use this repository, please cite the following [publication](https://arxiv.org/abs/2111.14465):

```bibtex
@inproceedings{mfb,
  title = {Motion-from-Blur: 3D Shape and Motion Estimation of Motion-blurred Objects in Videos},
  author = {Denys Rozumnyi and Martin R. Oswald and Vittorio Ferrari and Marc Pollefeys},
  booktitle = {CVPR},
  month = {Jun},
  year = {2022}
}
```
