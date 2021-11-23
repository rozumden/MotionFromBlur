Kaolin is available here: https://github.com/NVIDIAGameWorks/kaolin
Pre-trained DeFMO is implemented directly in kornia.feature.DeFMO

The code can be easily run by 'python run.py --input video.avi', e.g. check run.sh script.
The results will be written to the output folder.

To run on the fast moving object deblurring benchmark, we used: https://github.com/rozumden/fmo-deblurring-benchmark
For synthetic dataset generation, we used the publicly available implementation from DeFMO authors: https://github.com/rozumden/DeFMO/tree/master/renderer
