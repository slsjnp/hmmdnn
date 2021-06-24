import torch


def extract_model_state_dict(ckpt_path, prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        for k, v in checkpoint['state_dict'].items():
            if not k.startswith('model.'):
                continue
            k = k[6:]  # remove 'model.'
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    else:  # if it only has model weights
        for k, v in checkpoint.items():
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
    # epoch = checkpoint_['epoch']
    # loss = checkpoint_['loss']
    # return epoch, loss