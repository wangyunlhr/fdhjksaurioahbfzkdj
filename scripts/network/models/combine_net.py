import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import dztimer



class CombinedModel(nn.Module):
    def __init__(self, Flow4D_NK_onlyrestore, Flow4D_forflow):
        super(CombinedModel, self).__init__()
        self.Flow4D_NK_onlyrestore= Flow4D_NK_onlyrestore
        self.Flow4D_forflow = Flow4D_forflow
        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def forward(self, batch, training_flag):
        restore_output = self.Flow4D_NK_onlyrestore(batch, training_flag)
        # restore_all_dict = {'pc0s_restore': restore_pc0_all, \
        #             'pc0s_gt': batch["pc0_all"].reshape(batch_sizes,-1,3),
        #             'pc1s_restore': restore_pc1_all, \
        #             'pc1s_gt': batch["pc1_all"].reshape(batch_sizes,-1,3),
        #             }
        
        restore_pc0 = restore_output['pc0s_restore'].detach()
        restore_pc1 = restore_output['pc1s_restore'].detach()
        # restore_pc0 = torch.cat((restore_output['pc0s_restore'].detach(), batch['pc0']), dim = 1)
        # restore_pc1 = torch.cat((restore_output['pc1s_restore'].detach(), batch['pc1']), dim = 1)
        new_batch = {'pc0_all': restore_pc0, 'pc1_all': restore_pc1, 
                     'pose0': batch['pose0'], 'pose1': batch['pose1'],
                     'pc0': batch['pc0'], 'pc1': batch['pc1'],
                     'ego_motion': batch['ego_motion'] }

        flow_output = self.Flow4D_forflow(new_batch)
        return restore_output, flow_output