### preprocess dataset
---
数据处理的环境./dataprocess/envprocess.yaml 

./dataprocess/extract_dynamic_multi_frame_idx.py
1.需要修改累积帧数multi_frame  2.路径
```bash
python dataprocess/extract_dynamic_multi_frame_idx.py --av2_type sensor --data_mode train --argo_dir /data0/dataset/av2 --output_dir /data1/dataset/av2/debug2 --multi_frame 5
```
