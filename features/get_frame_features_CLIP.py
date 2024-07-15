import torch
import clip
from PIL import Image
import time
import glob
from tqdm import tqdm
import os

device = "cuda:7" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_rotate_degree(view):
    rotate_dict = {'ego_m':0, 
                'ego_r':0, 
                'ego_l':0, 
                'exo_r':270, 
                'exo_l':90,
                'exo_m':0}
    return rotate_dict[view]

def get_vid_frame_feature():
    image_tmpl='frame_{:010d}.jpg'

    with torch.no_grad():
        for record_path in tqdm(glob.glob('/PATH/TO/THE/FRAME/ROOT/*')):
            record = record_path.split('/')[-1]
            frame_root = '/PATH/TO/THE/FRAME/ROOT/{}'.format(record)
            views = glob.glob(os.path.join(frame_root, '*/'))
            for view in tqdm(views, leave=False):
                view_name = view.split('/')[-2]
                rotate_d = get_rotate_degree(view_name)
                img_features = torch.Tensor([]).to(device)
                num_frame = len(glob.glob(os.path.join(view, '*')))
                save_path = './Ego_FBAU_CLIP_Vid_Feat_w_Rotate/{}/{}/clip_vit_b32_vid_frame_feat.pth'.format(record, view_name)
                save_root = './Ego_FBAU_CLIP_Vid_Feat_w_Rotate/{}/{}/'.format(record, view_name)
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                if os.path.exists(save_path):
                    continue
                for frame_id in tqdm(range(1, num_frame+1), leave=False):
                    frame_name = image_tmpl.format(frame_id)
                    frame_path = os.path.join(view, frame_name)
                    image = preprocess(Image.open(frame_path).rotate(rotate_d, expand=True)).unsqueeze(0).to(device)
                    image_feature = model.encode_image(image)
                    img_features = torch.cat([img_features, image_feature], dim=0)
                
                img_features = img_features.detach().cpu()
                save_data = {'clip_feat':img_features, 'record':record, 'view':view_name}
                torch.save(save_data, save_path)

get_vid_frame_feature()
