# Local Attention Module
## Dataset
- Nyudv2
## Framework
- PyTorch <br/>
- Python 3.7 <br/>
## Task
- Segmentation 
## Testing Dataloder
- Run python3 eval.py
## Configuration(Config/configuration.py)
- Every path is relative. Path description is provided in the file

## Running the scripts
# To check MultiScale Attention
- go to: ./models/encoders.<br/>
- run python3 dual_segformer.py. It will run multiscale attention only (check **main** section). If you want to compare the output with Attention, call Attention too.<br/>
- to check how an input is modified by the whole Dual_Segformer Network, pass the input through that. You just need to change in the **main** section<br/>

## Acknowledgement

Our base code follows [RGBX Semantic Segmentation Repo](https://github.com/huaaaliu/RGBX_Semantic_Segmentation). The code is heavily based on [TorchSeg](https://github.com/ycszen/TorchSeg) and [SA-Gate](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch) too. Thank you everyone for their excellent work!

