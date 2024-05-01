#!/root/miniconda3/envs/GsE/bin/python
from dataclasses import dataclass

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import os
import glob
import json
import shutil
import matplotlib.pyplot as plt
import copy
import json
import subprocess
import gc
import mediapipe
import pandas as pd

# @threestudio.register("stable-diffusion-instructpix2pix-guidance")
class LadiVtonGuidance():
    # def openpose(self, im_tensor, keypoint_path='./output/keypoints/'):
    #     '''
    #     Extract human pose from input image.
    #     '''
    #     body_estimation = Body('/root/autodl-tmp/GaussianVTON/ladi/preprocess/pytorch_openpose/model/body_pose_model.pth')

    #     for i in range(im_tensor.size(0)):
    #         candidate, subset = body_estimation(im_tensor[i])
    #         canvas[i] = util.draw_bodypose(np.zeros_like(im_tensor[i]), candidate, subset)
    #         arr = candidate.tolist()
    #         vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    #         for i in range(0,18):
    #             if len(arr)==i or arr[i][3] != vals[i]:
    #                arr.insert(i, [-1, -1, -1, vals[i]])

    #         keypoints = {'keypoints':arr[:18]}
    #         with open(keypoint_path + i + ".json", "w") as fin:
    #             fin.write(json.dumps(keypoints))
        
    #     return canvas

    # def grayscale(self, im_tensor):
        
    #     mp ={0: 0, 128: 18, 64: 4, 132: 19, 69: 5, 136: 20, 75: 6, 140: 21, 145: 22, 85: 9, 150: 23, 90: 10, 155: 24, 121: 16, 105: 13, 111: 14, 52: 2, 117: 15, 57: 3, 124: 17,
    #          2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 9, 10: 10, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24}
        
    #     lut = np.zeros((256, 1), dtype = np.uint8)

    #     for i in range(0, 256):
    #         lut[i] = mp.get(i) or mp[min(mp.keys(), key = lambda key: abs(key-i))]

    #     for i in range(im_tensor.size(0)):
    #         image_np = im_tensor[i].numpy().astype(np.uint8)  # Convert tensor to NumPy array
    #         image_gray = cv2.LUT(image_np, lut)
    #         processed_images.append(torch.from_numpy(image_gray).unsqueeze(0))

    #     return torch.cat(processed_images, dim=0)
    
    # def resize_image_pad(self, im, target_width, target_height):
    #     '''
    #     Resize PIL image keeping ratio and using white background.
    #     '''
    #     target_ratio = target_height / target_width
    #     im_ratio = im.height / im.width
    #     if target_ratio > im_ratio:
    #         # It must be fixed by width
    #         resize_width = target_width
    #         resize_height = round(resize_width * im_ratio)
    #     else:
    #         # Fixed by height
    #         resize_height = target_height
    #         resize_width = round(resize_height / im_ratio)

    #     image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    #     background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    #     offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    #     background.paste(image_resize, offset)
    #     return background.convert('RGB')

    # def resize_with_pad(self, im_tensor, target_width, target_height):
    #     '''
    #     Resize PyTorch tensor keeping ratio and using white background.
    #     '''
    #     # Convert tensor to PIL Image
    #     im = F.to_pil_image(im_tensor)

    #     target_ratio = target_height / target_width
    #     im_ratio = im.height / im.width

    #     if target_ratio > im_ratio:
    #         # It must be fixed by width
    #         resize_width = target_width
    #         resize_height = round(resize_width * im_ratio)
    #     else:
    #         # Fixed by height
    #         resize_height = target_height
    #         resize_width = round(resize_height / im_ratio)

    #     # Resize using PIL
    #     image_resize = F.resize(im, (resize_width, resize_height), Image.ANTIALIAS)

    #     # Create a new RGBA image with white background
    #     background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))

    #     # Calculate offset for centering
    #     offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))

    #     # Paste resized image onto the background
    #     background.paste(image_resize, offset)

    #     # Convert result to RGB and back to PyTorch tensor
    #     result_tensor = F.to_tensor(background.convert('RGB'))

    #     return result_tensor
    
    # def resize_with_pad_path(self, im_tensors, target_width, target_height, output_path):
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)

    #     processed_paths = []

    #     def resize_pad(im_tensor, target_width, target_height):
    #         im = transforms.ToPILImage()(im_tensor)

    #         target_ratio = target_height / target_width
    #         im_ratio = im.height / im.width

    #         if target_ratio > im_ratio:
    #             resize_width = target_width
    #             resize_height = round(resize_width * im_ratio)
    #         else:
    #             resize_height = target_height
    #             resize_width = round(resize_height / im_ratio)

    #         image_resize = transforms.Resize((resize_width, resize_height))(im)
    #         background = transforms.ToPILImage()(transforms.ToTensor()(transforms.Resize((target_width, target_height))(image_resize)))

    #         offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    #         background.paste(image_resize, offset)

    #         return transforms.ToTensor()(background.convert('RGB'))

    #     for i, im_tensor in enumerate(im_tensors):
    #         result_tensor = resize_pad(im_tensor, target_width, target_height)

    #         # Save the processed image
    #         filename = f"{i}_0.png"
    #         output_file_path = os.path.join(output_path, filename)
    #         transforms.ToPILImage()(result_tensor).save(output_file_path)

    #         processed_paths.append(output_file_path)

    #     return processed_paths

    # def load_images_in_folder(self, folder_path):
    #     image_list = []
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(".png"):
    #             img_path = os.path.join(folder_path, filename)
    #             img = Image.open(img_path)
    #             image_list.append(np.array(img))
    #     return image_list

    # def create_tensor_from_images(self, image_list):
    #     # Assuming all images have the same size
    #     batch_size, height, width, channels = len(image_list), image_list[0].shape[0], image_list[0].shape[1], image_list[0].shape[2]
    #     image_tensor = np.zeros((batch_size, height, width, channels), dtype=np.uint8)

    #     for i, img_array in enumerate(image_list):
    #         image_tensor[i] = img_array

    #     return image_tensor

    # def get_cloth_mask(self, image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     mask = np.zeros_like(image)
    #     cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    #     return mask

    # def write_edge(self, C_path,E_path):
    #     img = cv2.imread(C_path)
    #     res = get_cloth_mask(img)
    #     if(np.mean(res)<100):
    #         ot = otsu(img,11,0.6)
    #         res = contour(ot)
    #     cv2.imwrite(E_path,res)

    def generate_test_pairs_txt(image_folder, prompt_path, output_file):
        # 获取prompt_path中的倒数第二个名字作为category
        category_name = os.path.basename(os.path.dirname(prompt_path))
        category_mapping = {'dresses': 2, 'lower_body': 1, 'upper_body': 0}
        category_value = category_mapping.get(category_name, -1)
        if category_value == -1:
            print("Invalid category value")
            return
    
        # 清空并重新写入txt文件
        with open(output_file, 'w') as f:
            # 获取所有图片文件名
            image_files = sorted(os.listdir(image_folder))
            
            # 循环遍历图片，写入txt文件
            for image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                line = f"{image_file} {os.path.basename(prompt_path)} {category_value}\n"
                f.write(line)
    
    def resize_with_pad(im, target_width, target_height):
        '''
        Resize PIL image keeping ratio and using white background.
        '''
        target_ratio = target_height / target_width
        im_ratio = im.height / im.width
        if target_ratio > im_ratio:
            # It must be fixed by width
            resize_width = target_width
            resize_height = round(resize_width * im_ratio)
        else:
            # Fixed by height
            resize_height = target_height
            resize_width = round(resize_height / im_ratio)

        image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
        offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
        background.paste(image_resize, offset)
        return background.convert('RGB')

        #Add pairs
    def write_row(file_, *columns):
        print(*columns, sep='\t', end='\n', file=file_)

    def resize_write_pad(im, target_width, target_height):
        '''
        Resize PIL image keeping ratio and using white background.
        '''
        target_ratio = target_height / target_width
        im_ratio = im.height / im.width
        if target_ratio > im_ratio:
            # It must be fixed by width
            resize_width = target_width
            resize_height = round(resize_width * im_ratio)
        else:
            # Fixed by height
            resize_height = target_height
            resize_width = round(resize_height / im_ratio)

        image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
        offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
        background.paste(image_resize, offset)
        return background.convert('RGB')

    def otsu(img , n  , x ):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(img_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,n,x)
        return thresh

    def contour(img):
        edges = cv2.dilate(cv2.Canny(img,200,255),None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
        return masked

    def get_cloth_mask(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        return mask

    def write_edge(C_path,E_path):
        img = cv2.imread(C_path)
        res = get_cloth_mask(img)
        if(np.mean(res)<100):
            ot = otsu(img,11,0.6)
            res = contour(ot)
        cv2.imwrite(E_path,res)
    
    def load_images_as_tensor(directory):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = Image.open(os.path.join(directory, filename))
                img = np.array(img)
                images.append(img)
        images_tensor = np.stack(images, axis=0)
        return images_tensor

    def __call__(
        self,
        human: Float[Tensor, "B H W C"],
        prompt_path: PromptProcessorOutput,
        **kwargs,
    ):
        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/input/*/*/*.*')
        for f in files:
            os.remove(f)

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/results/*/*/*.*')
        for f in files:
            os.remove(f)

        upper = open('/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/test_pairs_unpaired.txt', 'w')
        lower = open('/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/test_pairs_unpaired.txt', 'w')
        dresses = open('/root/autodl-tmp/GaussianVTON/ladi/input/dresses/test_pairs_unpaired.txt', 'w')
        all = open('/root/autodl-tmp/GaussianVTON/ladi/input/test_pairs_paired.txt', 'w')

        human_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans'
        if not os.path.exists(human_path):
            os.makedirs(output_folder)

        for i in range(human.shape[0]):
            image_array = human.shpae[i]  
            image_array = np.uint8(image_array)  
            image = Image.fromarray(image_array) 
            image_path = os.path.join(output_folder, f"{i+1:02d}_0.jpg")
            image.save(image_path)

        txt_path = '/root/autodl-tmp/GaussianVTON/ladi/images/test_pairs.txt'
        generate_test_pairs_txt(human_path, prompt_path, txt_path)

        with open('/root/autodl-tmp/GaussianVTON/ladi/images/test_pairs.txt', "r") as file:
            data = file.readlines()
            for line in data:
                word = line.split()
                org_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans/' + word[0]
                if(word[2] == '0'):
                    write_row(upper,'0'+word[0],word[1])
                    write_row(all,'0'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/images/0' + word[0]
                elif(word[2] == '1'):
                    write_row(lower,'1'+word[0],word[1])
                    write_row(all,'1'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/images/1' + word[0]
                elif(word[2] == '2'):
                    write_row(dresses,'2'+word[0],word[1])
                    write_row(all,'2'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/images/2' + word[0]
                image = Image.open(org_path)
                new = resize_with_pad(image,384,512)
                new.save(res_path)

        upper.close()
        lower.close()
        dresses.close()
        all.close()

        command = "/root/autodl-tmp/GaussianVTON/ladi/subprocess_1.py"
        subprocess.call(command, shell=True)

        command = "/root/miniconda3/envs/GsE/bin/python /root/autodl-tmp/GaussianVTON/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore '/root/autodl-tmp/GaussianVTON/ladi/preprocess/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/images/' --output-dir '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/label_maps/'"
        subprocess.call(command, shell=True)
        command = "/root/miniconda3/envs/GsE/bin/python /root/autodl-tmp/GaussianVTON/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore '/root/autodl-tmp/GaussianVTON/ladi/preprocess/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/images/' --output-dir '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/label_maps/'"
        subprocess.call(command, shell=True)
        command = "/root/miniconda3/envs/GsE/bin/python /root/autodl-tmp/GaussianVTON/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore '/root/autodl-tmp/GaussianVTON/ladi/preprocess/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/images/' --output-dir '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/label_maps/'"
        subprocess.call(command, shell=True)

        command = "/root/autodl-tmp/GaussianVTON/ladi/subprocess_2.py"
        subprocess.call(command, shell=True)

        pattern = '/root/autodl-tmp/GaussianVTON/ladi/input/*/dense/*'
        mp ={0: 0, 128: 18, 64: 4, 132: 19, 69: 5, 136: 20, 75: 6, 140: 21, 145: 22, 85: 9, 150: 23, 90: 10, 155: 24, 121: 16, 105: 13, 111: 14, 52: 2, 117: 15, 57: 3, 124: 17,
            2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 9, 10: 10, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24}

        lut = np.zeros((256, 1), dtype=np.uint8)

        for i in range(0, 256):
            lut[i] = mp.get(i) or mp[min(mp.keys(), key = lambda key: abs(key-i))]

        for images in glob.glob(pattern):
            if images.endswith(".png"):
                image = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(images, cv2.LUT(image, lut))

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/input/*/*/*.*')
        for f in files:
            if f.endswith("_1.jpg") or f.endswith("_1.png"):
                os.remove(f)

        for c in ['dresses','upper_body','lower_body']:
            files = glob.glob('images/'+c+'/*.*')
            path = 'input/' + c + '/images/'
            for f in files:
                if f.endswith("_1.jpg"):
                    res = path +os.path.basename(f)
                    shutil.copy (f, res)
                    image = Image.open(res)
                    new = resize_with_pad(image,384,512)
                    new.save(res)

        for s in ['upper_body','lower_body','dresses']:
            input_path = '/root/autodl-tmp/GaussianVTON/ladi/input/' + s + '/images/'
            output_path = '/root/autodl-tmp/GaussianVTON/ladi/input/'+ s + '/masks/'
            for images in glob.glob('*',root_dir = input_path):
                if images.endswith("_1.jpg"):
                    write_edge(input_path + images , output_path+ os.path.splitext(images)[0] +".png")
        
        gc.collect()
        command = "/root/miniconda3/envs/GsE/bin/python /root/autodl-tmp/GaussianVTON/ladi/src/inference.py --num_inference_steps 20 --dataset dresscode --dresscode_dataroot /root/autodl-tmp/GaussianVTON/ladi/input  --output_dir /root/autodl-tmp/GaussianVTON/ladi/results --test_order unpaired  --batch_size 3 --num_workers 2 --enable_xformers_memory_efficient_attention"
        subprocess.call(command, shell=True)

########## Refinement ##

        dresscode = 'final'
        filepath = os.path.join('/root/autodl-tmp/GaussianVTON/ladi/input', f"test_pairs_paired.txt")
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        org_paths = sorted(
            [os.path.join('/root/autodl-tmp/GaussianVTON/ladi/input',category, 'images', line.strip().split()[0]) for line in lines for category in['lower_body', 'upper_body', 'dresses'] if
            os.path.exists(os.path.join('/root/autodl-tmp/GaussianVTON/ladi/input',category, 'images', line.strip().split()[0]))]
        )
        res_paths = sorted(
            [os.path.join('/root/autodl-tmp/GaussianVTON/ladi/results/unpaired', category, name) for category in ['lower_body', 'upper_body', 'dresses'] for name in os.listdir(os.path.join('/root/autodl-tmp/GaussianVTON/ladi/results/unpaired', category)) if
            os.path.exists(os.path.join('/root/autodl-tmp/GaussianVTON/ladi/results/unpaired', category, name))]
        )
        
        assert len(org_paths) == len(res_paths)
        sz = len(org_paths)

        for iter in range(0, sz):
            org_img = cv2.imread(org_paths[iter])
            org_res = cv2.imread(res_paths[iter])
            h, w = int(org_img.shape[0]/2), org_img.shape[1]
            img = org_img[:h, :w]
            res = org_res[:h, :w]
            mp_face_mesh = mediapipe.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode = True)
            results = face_mesh.process(img[:, :, ::-1])
            if(results.multi_face_landmarks == None):
                print('miss')
                continue
            landmarks = results.multi_face_landmarks[0]
            df = pd.DataFrame(list(mp_face_mesh.FACEMESH_FACE_OVAL), columns = ['p1', 'p2'])
            routes_idx = []

            p1 = df.iloc[0]['p1']
            p2 = df.iloc[0]['p2']
            for i in range(0, df.shape[0]):
                obj = df[df['p1'] == p2]
                p1 = obj['p1'].values[0]
                p2 = obj['p2'].values[0]

                cur = []
                cur.append(p1)
                cur.append(p2)
                routes_idx.append(cur)

            routes = []
            for sid, tid in routes_idx:
                sxy = landmarks.landmark[sid]
                txy = landmarks.landmark[tid]

                source = (int(sxy.x * img.shape[1]), int(sxy.y * img.shape[0]))
                target = (int(txy.x * img.shape[1]), int(txy.y * img.shape[0]))

                routes.append(source)
                routes.append(target)

            mask = np.zeros((img.shape[0], img.shape[1]))
            mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
            mask = mask.astype(bool)
            res[mask] = img[mask]
            org_img[:h, :w] = img
            org_res[:h, :w] = res
            cv2.imwrite(res_paths[iter].replace('/root/autodl-tmp/GaussianVTON/ladi/results/unpaired', 'final').replace('_0.jpg', '_' + dresscode + '.jpg'), org_res)  

        category_name = os.path.basename(os.path.dirname(prompt_path))
        final_path = '/root/autodl-tmp/GaussianVTON/ladi/final'
        final_path = os.path.join(final_path, category_name)
        final_tensor = load_images_as_tensor(final_path)

        return final_tensor
        # pad_path = '/root/autodl-tmp/GaussianVTON/pad_human/'

        # human_tensor = resize_with_pad(human, 384, 512) #resize with pad
        # pad_path = resize_with_pad_path(human, 384, 512, pad_path) #resize with pad
        # human_pose = openpose(human_tensor, ) #extrate pose
        # human_label = parsing(human_path, dataset='atr', model_restore='/root/autodl-tmp/GaussianVTON/ladi/preprocess/Self_Correction_Human_Parsing/checkpoints/final.pth')

        # command_list = [
        #     'python', 'ladi/preprocess/detectron2/projects/DensePose/apply_net.py', 'show',
        #     'ladi/preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        #     'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
        #     '/root/autodl-tmp/GaussianVTON/pad_human',
        #     'dp_segm', '-v',
        #     '--output', '/root/autodl-tmp/GaussianVTON/dense_human/'
        # ]
        # subprocess.run(command_list, check=True)

        # command_list = [
        #     'python', 'ladi/preprocess/detectron2/projects/DensePose/apply_net.py', 'dump',
        #     'ladi/preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        #     'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
        #     '/root/autodl-tmp/GaussianVTON/pad_human',
        #     'dp_segm', '-v',
        #     '--output', '/root/autodl-tmp/GaussianVTON/dense_human/'
        # ]
        # subprocess.run(command_list, check=True)

        # image_list = load_images_in_folder('/root/autodl-tmp/GaussianVTON/dense_human/')
        # human_dense = create_tensor_from_images(image_list) 
        # human_dense = grayscale(human_dense)

        # cloth_path = '/root/autodl-tmp/GaussianVTON/pad_cloth/'
        # mask_cloth_path = '/root/autodl-tmp/GaussianVTON/mask_cloth/'

        # # files = glob.glob(pad_cloth_path+'/*.*')
        # pad_cloth_path = cloth_path + os.path.basename(prompt_path)
        # cloth_image = Image.open(prompt_path)
        # pad_cloth = resize_image_pad(cloth_image, 384, 512)
        # pad_cloth.save(pad_cloth_path)

        # for images in glob.glob('*', root_dir = cloth_path):
        #     if images.endswith(".jpg"):
        #         write_edge(pad_cloth_path, mask_cloth_path+ os.path.splitext(images)[0] + ".png")
