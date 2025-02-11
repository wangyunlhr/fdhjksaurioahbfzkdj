### preprocess dataset
---
数据处理的环境./dataprocess/envprocess.yaml 

数据处理代码./dataprocess/extract_dynamic_multi_frame_idx.py
1.需要修改累积帧数multi_frame(奇数）  2.路径
```bash
train
python dataprocess/extract_dynamic_multi_frame_idx.py --av2_type sensor --data_mode train --argo_dir /data0/dataset/av2 --output_dir /data1/dataset/av2/debug2 --multi_frame 5
```
```
val/test 添加mask路径
python dataprocess/extract_dynamic_multi_frame_idx.py --av2_type sensor --data_mode val --argo_dir /data0/dataset/av2 --output_dir /data1/dataset/av2/debug --multi_frame 5 --mask_dir /data0/dataset/av2/eval_mask
```
