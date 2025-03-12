class HDF5Dataset_after(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_after, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('./conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        # ! eval_all image
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        #! eval all
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if (self.scene_id_bounds[scene_id]["max_index"]-1) <= index_: 
                index_ = index_ - 2
        scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            pc0 = torch.tensor(f[key]['lidar'][:]) 
            gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 

            if self.scene_id_bounds[scene_id]["max_index"] == index_:
                print("!!!!!!__getitem__(index_ + 1)") 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                after_frames = []
                num_past_frames = (self.n_frames - 2)//2  
                num_after_frames = self.n_frames - 2 - num_past_frames

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

                for i_a in range(2, num_after_frames + 2):
                    frame_index_a = index_ + i_a
                    if frame_index_a > self.scene_id_bounds[scene_id]["max_index"]: 
                        frame_index_a = self.scene_id_bounds[scene_id]["max_index"] 

                    after_timestamp = str(self.data_index[frame_index_a][1])
                    after_pc = torch.tensor(f[after_timestamp]['lidar'][:])
                    after_gm = torch.tensor(f[after_timestamp]['ground_mask'][:])
                    after_pose = torch.tensor(f[after_timestamp]['pose'][:])

                    after_frames.append((after_pc, after_gm, after_pose))

                for i, (after_pc, after_gm, after_pose) in enumerate(after_frames):
                    res_dict[f'pc_a{i+1}'] = after_pc
                    res_dict[f'gm_a{i+1}'] = after_gm
                    res_dict[f'pose_a{i+1}'] = after_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices #原始的category属性
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor #映射之后的label

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

            #! eval all
            # res_dict['eval_mask'] = (~(gm0 | (torch.tensor(f[key]['flow_category_indices'][:]) == 0)))
            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict


def collate_fn_pad_after(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    num_frames_after = 2
    while f'pc_a{num_frames_after - 1}' in batch[0]:
        num_frames_after += 1

    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground= [], []
    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    pc_a_after_mask_ground = [[] for _ in range(num_frames_after - 2)]
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['pc0'][~batch[i]['gm0']])
        pc1_after_mask_ground.append(batch[i]['pc1'][~batch[i]['gm1']])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
        for j in range(1, num_frames_after - 1):
            pc_a_after_mask_ground[j-1].append(batch[i][f'pc_a{j}'][~batch[i][f'gm_a{j}']])
    

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]
    pc_a_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_a, batch_first=True, padding_value=torch.nan) for pc_a in pc_a_after_mask_ground]


    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    for j in range(1, num_frames_after - 1):
        res_dict[f'pc_a{j}'] = pc_a_after_mask_ground[j-1]
        res_dict[f'pose_a{j}'] = [batch[i][f'pose_a{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict

#NOTE Flow4D.py part
        num_frames_before = (self.num_frames - 2) // 2
        num_frames_after = (self.num_frames - 2) - num_frames_before
        transform_pc_m_frames = [[] for _ in range(num_frames_before)]
        transform_pc_a_frames = [[] for _ in range(num_frames_after)]


        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id] 
            self.timer[0][0].start("pose")
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id] 
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id]) 

                if num_frames_before > 0: 
                    past_poses = []
                    for i in range(1, num_frames_before + 1):
                        past_pose = cal_pose0to1(batch[f"pose_m{i}"][batch_id], batch["pose1"][batch_id])
                        past_poses.append(past_pose)

                if num_frames_after > 0: 
                    after_poses = []
                    for i in range(1, num_frames_after + 1):
                        after_pose = cal_pose0to1(batch[f"pose_a{i}"][batch_id], batch["pose1"][batch_id])
                        after_poses.append(after_pose)
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3] #t -> t+1 warping
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)
            transform_pc0s.append(transform_pc0)

            for i in range(1, num_frames_before + 1):
                selected_pc_m = batch[f"pc_m{i}"][batch_id]
                transform_pc_m = selected_pc_m @ past_poses[i-1][:3, :3].T + past_poses[i-1][:3, 3]
                transform_pc_m_frames[i-1].append(transform_pc_m)
            
            for i in range(1, num_frames_after + 1):
                selected_pc_a = batch[f"pc_a{i}"][batch_id]
                transform_pc_a = selected_pc_a @ after_poses[i-1][:3, :3].T + after_poses[i-1][:3, 3]
                transform_pc_a_frames[i-1].append(transform_pc_a)

        pc_m_frames = [torch.stack(transform_pc_m_frames[i], dim=0) for i in range(num_frames_before)]
        pc_a_frames = [torch.stack(transform_pc_a_frames[i], dim=0) for i in range(num_frames_after)]

        pc0s = torch.stack(transform_pc0s, dim=0) 
        pc1s = batch["pc1"]
        self.timer[0].stop()


        pcs_dict = {
            'pc0s': pc0s,
            'pc1s': pc1s, #! change_to_pc1s_gt
        }
        for i in range(1, num_frames_before + 1):
            pcs_dict[f'pc_m{i}s'] = pc_m_frames[i-1]

        for i in range(1, num_frames_after + 1):
            pcs_dict[f'pc_a{i}s'] = pc_a_frames[i-1]


        self.timer[1].start("4D_voxelization")
        dict_4d = self.embedder_4D(pcs_dict)
        pc01_tesnor_4d = dict_4d['4d_tensor']
        pc0_3dvoxel_infos_lst =dict_4d['pc0_3dvoxel_infos_lst']
        pc0_point_feats_lst =dict_4d['pc0_point_feats_lst']
        pc0_num_voxels = dict_4d['pc0_mum_voxels']
        self.timer[1].stop()

        self.timer[2].start("4D_backbone")
        pc_all_output_4d = self.network_4D(pc01_tesnor_4d) #all = past, current, next 다 합친것
        self.timer[2].stop()
