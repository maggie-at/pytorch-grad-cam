import os
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import models, transforms

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cnn_cam(data_path, res_path, num_classes):
    model = models.resnet34()
    # 修改layer4的输出特征图尺寸
    # model.layer4[0].conv1.stride = (1, 1)
    # model.layer4[0].downsample[0].stride = (1, 1)
    # print(model)
    # return 
    in_channel = model.fc.in_features
    model.fc = torch.nn.Linear(in_channel, num_classes)
    model.load_state_dict(torch.load('./train_model/resnet/resNet34-dagm.pth', map_location=device))
    
    # LayerCam & WeightCam替换一下utils文件
    
    target_layers = [model.layer2, model.layer3, model.layer4]

    # GradCam
    # target_layers = [model.layer4[2].conv2]


    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    
    cls_folder_path = os.path.join(data_path, "defect/")
    # 在结果目录中创建一个与类别相同的文件夹
    res_folder_path = res_path
    os.makedirs(res_folder_path, exist_ok=True)
    os.makedirs(os.path.join(res_folder_path, "gray/"), exist_ok=True)
    os.makedirs(os.path.join(res_folder_path, "bbox/"), exist_ok=True)
    os.makedirs(os.path.join(res_folder_path, "label/"), exist_ok=True)

    targets = [ClassifierOutputTarget(0)]

   
    for img_file in tqdm(os.listdir(cls_folder_path), desc="Processing: "):
        img_path = os.path.join(cls_folder_path, img_file)
        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # [C, H, W]
        img_tensor = data_transform(img)
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
        plt.imsave(os.path.join(res_folder_path, "gray/", img_file), visualization)
        
        grayscale_cam_8bit = (grayscale_cam * 255).astype('uint8')
        
        # 设置每个类别的分割阈值
        _, thresh = cv2.threshold(grayscale_cam_8bit, 127, 255, cv2.THRESH_BINARY)
        
        # 画出预测框
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            # 忽略面积小于64的预测框
            if area >= 64:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(res_folder_path, "bbox/", img_file), img)

        # save bbox coordinates to txt file
        with open(os.path.join(res_folder_path, "label/", img_file.split('.')[0] + '.txt'), 'w') as f:
            for contour in contours:
                # 忽略面积小于64的预测框
                if area >= 49:
                    x, y, w, h = cv2.boundingRect(contour)
                    f.write(f"{x} {y} {x+w} {y+h}\n")


if __name__ == '__main__':
    cnn_cam(data_path='/home/ubuntu/workspace/hy/dataset/DAGM/val/', 
            res_path='/home/ubuntu/workspace/hy/dataset/DAGM/result/predict_layer_234/',
            num_classes=2)


# 跑一下Full / Score / Ablation，看看效果

# 走之前跑上Full，明早看看效果