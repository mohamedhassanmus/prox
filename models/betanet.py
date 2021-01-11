import torch
import torch.nn as nn

import sys


class FC(nn.Module):
    def __init__(self):

        super(FC, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################

        self.count = 0

        # self.FC_pack_64 = nn.Sequential(
        #     nn.Linear(12, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 2),
        #     nn.ReLU(inplace=True),
        # )

        self.FC_pack_64 = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor

        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor

        self.dtype = dtype

    def forward(self, batch_gender, batch_betas):
        # batch gender is B x 2. use [1, 0] for f and [0, 1] for m
        # batch betas is B x 10. SMPL body shape parameters in order

        batch_wtht = self.FC_pack_64(torch.cat((batch_gender, batch_betas), 1))

        batch_weight_kg_est = batch_wtht[:, 0:1]
        batch_height_m_est = batch_wtht[:, 1:2]

        return batch_weight_kg_est, batch_height_m_est


if __name__ == "__main__":
    # Super hacky shit to convert a pytorch 1.6 model checkpoint into a 1.0 file

    def load_model_txt(model, path):
        data_dict = {}
        fin = open(path, 'r')
        i = 0
        odd = 1
        prev_key = None
        while True:
            s = fin.readline().strip()
            if not s:
                break
            if odd:
                prev_key = s
            else:
                print('Iter', i)
                val = eval(s)
                if type(val) != type([]):
                    data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
                else:
                    data_dict[prev_key] = torch.FloatTensor(eval(s))
                i += 1
            odd = (odd + 1) % 2

        # Replace existing values with loaded

        print('Loading...')
        own_state = model.state_dict()
        print('Items:', len(own_state.items()))
        for k, v in data_dict.items():
            if not k in own_state:
                print('Parameter', k, 'not found in own_state!!!')
            else:
                try:
                    own_state[k].copy_(v)
                except:
                    print('Key:', k)
                    print('Old:', own_state[k])
                    print('New:', v)
                    sys.exit(0)
        print('Model loaded')

    def save_model_txt(model, path):
        fout = open(path, 'w')
        for k, v in model.state_dict().items():
            fout.write(str(k) + '\n')
            fout.write(str(v.tolist()) + '\n')
        fout.close()

    # Save
    # model = torch.load('models/betanet_twolayer.pt', map_location=torch.device('cpu'))
    # save_model_txt(model, 'models/betanet_text.txt')

    # Load
    model = FC()
    load_model_txt(model, 'models/betanet_text.txt')
    torch.save(model, 'models/betanet_old_pytorch.pt')
