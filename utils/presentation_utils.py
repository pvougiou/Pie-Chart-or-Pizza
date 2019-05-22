#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#

from architecture import SimpleVGG16
import requests
import os
import cPickle as pickle
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

params= {'init_weight': 1e-3,
         'dropout': 0.5,
         'device': torch.device("cpu")}

# Set the location of the pre-trained model checkpoint.
checkpoint_dir = './checkpoints/model.num_classes_12.best.pth.tar'

train_images_stats_dir = './data/stats.p'
idx_to_class = {0: 'AreaGraph', 1: 'BarChart', 2: 'BoxPlot', 3: 'LineGraph', 4: 'Map', 5: 'ParetoChart', 6: 'PieChart', 7: 'RadarPlot', 8: 'ScatterGraph', 9: 'Table', 10: 'Various', 11: 'VennDiagram'}

with open(train_images_stats_dir, 'rb') as f:
    stats_file = pickle.load(f)
    mean_dim1, mean_dim2, mean_dim3 = stats_file['mean']
    std_dim1, std_dim2, std_dim3 = stats_file['std']

    
normalize = transforms.Normalize(mean=[mean_dim1.item(), mean_dim2.item(), mean_dim3.item()],
                                 std=[std_dim1.item(), std_dim2.item(), std_dim3.item()])

test_data_transform = transforms.Compose([
    transforms.Resize(192),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    normalize
])


def download_image(image_url, target_file_location):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(target_file_location, 'wb') as handle:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
            handle.close()
    except Exception as e:
        print(e)
        os.remove(target_file_location)
    
    tmp_image = Image.open(target_file_location)
    plt.imshow(np.asarray(tmp_image))

def return_tensor(image_location):
    returned_image_tensor = []
    tmp_image = Image.open(image_location)
    plt.imshow(np.asarray(tmp_image))
    tmp_image = tmp_image.convert('RGB')
    tmp_image_tensor = test_data_transform(tmp_image)
    returned_image_tensor.append(tmp_image_tensor)
    returned_image_tensor.append(tmp_image_tensor)
    tmp_image.close()
    
    return torch.stack(returned_image_tensor)

def load_pretrained_model():
    cnn_network_with_pred_layer = SimpleVGG16(params, num_classes=len(idx_to_class.keys()))

    pre_trained_cnn_checkpoint = torch.load(checkpoint_dir, map_location=params['device'])
    cnn_network_state_dict = cnn_network_with_pred_layer.state_dict()
    # Filter out unnecessary keys from the pre-trained ConvNet.
    pretrained_dict = {k: v for k, v in pre_trained_cnn_checkpoint['cnn_state_dict'].items() if k in cnn_network_state_dict}
    # Overwrite entries in the existing state_dict.
    cnn_network_state_dict.update(pretrained_dict)
    # Load the new state dict.
    cnn_network_with_pred_layer.load_state_dict(pretrained_dict)
    cnn_network_with_pred_layer.eval()
    return cnn_network_with_pred_layer, idx_to_class
