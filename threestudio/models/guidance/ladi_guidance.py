from dataclasses import dataclass

from PIL import Image
import cv2
import numpy as np
import torch
# import torch.nn.functional as F
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
from threestudio.utils.base import BaseObject

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import torchvision.utils as vutils

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *

import lang_segment_anything
from lang_segment_anything.lang_sam import LangSAM

class LadiVtonGuidance:
    @torch.cuda.amp.autocast(enabled=False)
    def dhash(self, image, hash_size=8):
        image = image.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
        pixels = list(image.getdata())
        difference = [pixels[i * hash_size + j + 1] < pixels[i * hash_size + j] for i in range(hash_size) for j in range(hash_size)]
        decimal_value = 0
        hash_code = ""
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2 ** index
        hash_code = hex(decimal_value)[2:]
        return hash_code
    
    @torch.cuda.amp.autocast(enabled=False)
    def hamming_distance(self, hash1, hash2):
        return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
    
    @torch.cuda.amp.autocast(enabled=False)
    def calculate_dhash(self, image_path):
        image = Image.open(image_path)
        hash_code = self.dhash(image)
        return hash_code
    
    @torch.cuda.amp.autocast(enabled=False)
    def find_most_different_images(self, folder_path):
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

        most_different_images = []

        for i in range(0, len(image_paths), 3):
            if i + 2 >= len(image_paths):
                break
            
            hash_codes = [self.calculate_dhash(image_paths[j]) for j in range(i, i + 3)]

            similarities = []
            for j in range(3):
                for k in range(j + 1, 3):
                    hash1 = hash_codes[j]
                    hash2 = hash_codes[k]
                    distance = self.hamming_distance(hash1, hash2)
                    similarity = 1 - distance / (64 * 1.0)  
                    similarities.append((j, k, similarity))

            avg_similarities = []
            for j in range(3):
                other_similarities = [similarity for similarity in similarities if j in similarity[:2]]
                avg_similarity = np.mean([similarity[2] for similarity in other_similarities])
                avg_similarities.append(avg_similarity)

            min_avg_similarity_index = np.argmin(avg_similarities)
            most_different_image_index = min_avg_similarity_index + i
            most_different_image = os.path.basename(image_paths[most_different_image_index])
            most_different_images.append(most_different_image)

        return most_different_images


    @torch.cuda.amp.autocast(enabled=False)
    def crop_and_resize_image(self, image_path, middle_width, middle_height):
        image = Image.open(image_path)
        print(image.size)
        
        width, height = image.size
        left = (width - middle_width) // 2
        top = (height - middle_height) // 2
        right = left + middle_width
        bottom = top + middle_height
        
        cropped_image = image.crop((left, top, right, bottom))
        
        return cropped_image
    @torch.cuda.amp.autocast(enabled=False)
    def load_images_tensor(self, folder_path):
        image_paths = sorted(os.listdir(folder_path))
        images = []

        for image_path in image_paths:
            image = Image.open(os.path.join(folder_path, image_path))
            width, height = image.size
            image = np.array(image)  
            image = image / 255.0
            image_tensor = torch.tensor(image, dtype=torch.float32)
            images.append(image_tensor)

        return torch.stack(images, dim=0)


    @torch.cuda.amp.autocast(enabled=False)
    def generate_test_pairs_txt(self, image_folder, prompt_path, output_file):
        category_name = os.path.basename(os.path.dirname(prompt_path))
        category_mapping = {'dresses': 2, 'lower_body': 1, 'upper_body': 0}
        category_value = category_mapping.get(category_name, -1)
        if category_value == -1:
            print("Invalid category value")
            return
    
        with open(output_file, 'w') as f:
            image_files = sorted(os.listdir(image_folder))
            for image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                line = f"{image_file} {os.path.basename(prompt_path)} {category_value}\n"
                f.write(line)
    
    @torch.cuda.amp.autocast(enabled=False)
    def resize_with_pad(self, im, target_width, target_height):
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
    @torch.cuda.amp.autocast(enabled=False)
    def write_row(self, file_, *columns):
        print(*columns, sep='\t', end='\n', file=file_)

    @torch.cuda.amp.autocast(enabled=False)
    def resize_write_pad(self, im, target_width, target_height):
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

    @torch.cuda.amp.autocast(enabled=False)
    def otsu(self, img , n  , x ):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(img_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,n,x)
        return thresh

    @torch.cuda.amp.autocast(enabled=False)
    def contour(self, img):
        edges = cv2.dilate(cv2.Canny(img,200,255),None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
        return masked

    @torch.cuda.amp.autocast(enabled=False)
    def get_cloth_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        return mask

    @torch.cuda.amp.autocast(enabled=False)
    def write_edge(self, C_path,E_path):
        img = cv2.imread(C_path)
        res = self.get_cloth_mask(img)
        if(np.mean(res)<100):
            ot = self.otsu(img,11,0.6)
            res = self.contour(ot)
        cv2.imwrite(E_path,res)
    
    @torch.cuda.amp.autocast(enabled=False)
    def load_images_as_tensor(self, directory):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = Image.open(os.path.join(directory, filename))
                img = np.array(img)
                images.append(img)
        images_tensor = np.stack(images, axis=0)
        return images_tensor

    @torch.cuda.amp.autocast(enabled=False)
    def ladi_vton(self, sam_list, prompt_path, final_path, category_name, H, W):
        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/images/humans/*.*')
        for f in files:
            os.remove(f)

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/input/*/*/*.*')
        for f in files:
            os.remove(f)

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/{category_name}/*.*')
        for f in files:
            os.remove(f)

        upper = open('/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/test_pairs_unpaired.txt', 'w')
        lower = open('/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/test_pairs_unpaired.txt', 'w')
        dresses = open('/root/autodl-tmp/GaussianVTON/ladi/input/dresses/test_pairs_unpaired.txt', 'w')
        all = open('/root/autodl-tmp/GaussianVTON/ladi/input/test_pairs_paired.txt', 'w')

        human_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans'

        for filename in sam_list:
            source_path = os.path.join(final_path, filename)
            target_filename = filename.replace('0', '', 1)
            target_path = os.path.join(human_path, target_filename)

            shutil.copy(source_path, target_path)

        txt_path = '/root/autodl-tmp/GaussianVTON/ladi/images/test_pairs.txt'

        self.generate_test_pairs_txt(human_path, prompt_path, txt_path)

        with open('/root/autodl-tmp/GaussianVTON/ladi/images/test_pairs.txt', "r") as file:
            data = file.readlines()
            for line in data:
                word = line.split()
                org_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans/' + word[0]
                if(word[2] == '0'):
                    self.write_row(upper,'0'+word[0],word[1])
                    self.write_row(all,'0'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/images/0' + word[0]
                elif(word[2] == '1'):
                    self.write_row(lower,'1'+word[0],word[1])
                    self.write_row(all,'1'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/images/1' + word[0]
                elif(word[2] == '2'):
                    self.write_row(dresses,'2'+word[0],word[1])
                    self.write_row(all,'2'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/images/2' + word[0]
                image = Image.open(org_path)
                new = self.resize_with_pad(image,384,512)
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
            files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/images/'+c+'/*.*')
            path = '/root/autodl-tmp/GaussianVTON/ladi/input/' + c + '/images/'
            for f in files:
                if f.endswith("_1.jpg"):
                    res = path +os.path.basename(f)
                    shutil.copy (f, res)
                    image = Image.open(res)
                    new = self.resize_with_pad(image,384,512)
                    new.save(res)

        for s in ['upper_body','lower_body','dresses']:
            input_path = '/root/autodl-tmp/GaussianVTON/ladi/input/' + s + '/images/'
            output_path = '/root/autodl-tmp/GaussianVTON/ladi/input/'+ s + '/masks/'
            pattern = os.path.join(input_path, '*')
            # for images in glob.glob('*',root_dir = input_path):
            for images in glob.glob(pattern):
                images = os.path.basename(images)
                if images.endswith("_1.jpg"):
                    self.write_edge(input_path + images , output_path+ os.path.splitext(images)[0] +".png")
        
        gc.collect()
        command = "/root/miniconda3/envs/GsE/bin/python /root/autodl-tmp/GaussianVTON/ladi/src/inference.py --num_inference_steps 20 --dataset dresscode --dresscode_dataroot /root/autodl-tmp/GaussianVTON/ladi/input  --output_dir /root/autodl-tmp/GaussianVTON/ladi/results_sam --test_order unpaired  --batch_size 3 --num_workers 2 --enable_xformers_memory_efficient_attention"
        # command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/src/inference.py --num_inference_steps 20 --dataset dresscode --dresscode_dataroot ./input  --output_dir ./results --test_order unpaired  --batch_size 3 --num_workers 2 --enable_xformers_memory_efficient_attention"
        subprocess.call(command, shell=True)

        filepath = os.path.join('/root/autodl-tmp/GaussianVTON/ladi/input', f"test_pairs_paired.txt")
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        org_paths = sorted(
            [os.path.join('/root/autodl-tmp/GaussianVTON/ladi/input',category, 'images', line.strip().split()[0]) for line in lines for category in['lower_body', 'upper_body', 'dresses'] if
            os.path.exists(os.path.join('/root/autodl-tmp/GaussianVTON/ladi/input',category, 'images', line.strip().split()[0]))]
        )
        res_paths = sorted(
            [os.path.join('/root/autodl-tmp/GaussianVTON/ladi/results_sam/unpaired', category, name) for category in ['lower_body', 'upper_body', 'dresses'] for name in os.listdir(os.path.join('/root/autodl-tmp/GaussianVTON/ladi/results_sam/unpaired', category)) if
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
            cv2.imwrite(res_paths[iter].replace('results_sam', 'results'), org_res)
            # cv2.imwrite(res_paths[iter], org_res)  

        final_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired'
        img_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/final'
        final_path = os.path.join(final_path, category_name)

        for filename in os.listdir(final_path):
            
            filepath = os.path.join(final_path, filename)

            target_size = (W, H)
            img = self.crop_and_resize_image(filepath, W, H)

            img_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/final'

            imgpath = os.path.join(img_path, filename)

            img.save(imgpath)
        
        return

    def __call__(
        self,
        rendering: Float[Tensor, "B H W C"],
        human: Float[Tensor, "B H W C"],
        prompt_path,
        **kwargs,
    ):
        img_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/final'
        image_out = glob.glob(os.path.join(img_path, '*.jpg'))
        
        if image_out:
            img_tensor = self.load_images_tensor(img_path)
            # print("unresized huamn size:", img_tensor.shape)
            # bt, H, W, C = human.shape
            # target_shape = (H, W)
            # # resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='bilinear', align_corners=False, dim=1)
            # resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='area', align_corners=False, dim=1)
            # print("resized huamn size:", resized_tensor.shape)
            return {"edit_images": img_tensor}

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/images/humans/*.*')
        for f in files:
            os.remove(f)

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/input/*/*/*.*')
        for f in files:
            os.remove(f)

        files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/results/*/*/*.*')
        for f in files:
            os.remove(f)
        
        # files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/final/*/*.*')
        # for f in files:
        #     os.remove(f)

        upper = open('/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/test_pairs_unpaired.txt', 'w')
        lower = open('/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/test_pairs_unpaired.txt', 'w')
        dresses = open('/root/autodl-tmp/GaussianVTON/ladi/input/dresses/test_pairs_unpaired.txt', 'w')
        all = open('/root/autodl-tmp/GaussianVTON/ladi/input/test_pairs_paired.txt', 'w')

        human_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans'
        if not os.path.exists(human_path):
            os.makedirs(human_path)
        
        # print(rendering.shape)
        # print(human.shape)

        # human_np = np.array(human)
        # human_np = ((human_np - human_np.min()) / (human_np.max() - human_np.min()) * 255).astype(np.uint8)
        print("human:", human.shape)
        # image = Image.fromarray(human_np)
        # image = human[0]    
        # image = human[0].byte().cpu().numpy().transpose(1, 2, 0)
        # image_data = human[0].cpu().numpy().astype(np.uint8)  
        # image = Image.fromarray(image_data)  
        batchsize =human.shape[0]
        for i in range(batchsize):
            image = human[i]
            image_data = (image.cpu().numpy() * 255).astype(np.uint8)
            save_path = f'/root/autodl-tmp/GaussianVTON/ladi/images/humans/{i+1:02d}_0.jpg'
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR) 
            cv2.imwrite(save_path, image_data)

        # save_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans/01_0.jpg'
        # image.save(save_path)
        # vutils.save_image(image, save_path)

        # for i, image in enumerate(human):
        #     image = (image.cpu().numpy() * 255).astype('uint8')
        #     # image = (image * 255).astype('uint8')

        #     file_name = f"{i+1:02d}_0.jpg"  
        #     file_path = os.path.join(human_path, file_name)
        #     cv2.imwrite(file_path, image)
        # for i in range(human.shape[0]):
        #     image_array = human[i]  
        #     imagepath = os.path.join(human_path, f"{i+1:02d}_0.jpg")
        #     cv2.imwrite(imagepath, image_array)
        #     # image_array = np.uint8(image_array)  
        #     # image = Image.fromarray(image_array) 
        #     # image_path = os.path.join(human_path, f"{i+1:02d}_0.jpg")
        #     # image_array.save(imagepath)

        txt_path = '/root/autodl-tmp/GaussianVTON/ladi/images/test_pairs.txt'
        # print(prompt_path)
        # print(txt_path)
        self.generate_test_pairs_txt(human_path, prompt_path, txt_path)

        with open('/root/autodl-tmp/GaussianVTON/ladi/images/test_pairs.txt', "r") as file:
            data = file.readlines()
            for line in data:
                word = line.split()
                org_path = '/root/autodl-tmp/GaussianVTON/ladi/images/humans/' + word[0]
                if(word[2] == '0'):
                    self.write_row(upper,'0'+word[0],word[1])
                    self.write_row(all,'0'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/images/0' + word[0]
                elif(word[2] == '1'):
                    self.write_row(lower,'1'+word[0],word[1])
                    self.write_row(all,'1'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/images/1' + word[0]
                elif(word[2] == '2'):
                    self.write_row(dresses,'2'+word[0],word[1])
                    self.write_row(all,'2'+word[0],word[1],word[2])
                    res_path = '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/images/2' + word[0]
                image = Image.open(org_path)
                new = self.resize_with_pad(image,384,512)
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
            files = glob.glob('/root/autodl-tmp/GaussianVTON/ladi/images/'+c+'/*.*')
            path = '/root/autodl-tmp/GaussianVTON/ladi/input/' + c + '/images/'
            for f in files:
                if f.endswith("_1.jpg"):
                    res = path +os.path.basename(f)
                    shutil.copy (f, res)
                    image = Image.open(res)
                    new = self.resize_with_pad(image,384,512)
                    new.save(res)

        for s in ['upper_body','lower_body','dresses']:
            input_path = '/root/autodl-tmp/GaussianVTON/ladi/input/' + s + '/images/'
            output_path = '/root/autodl-tmp/GaussianVTON/ladi/input/'+ s + '/masks/'
            pattern = os.path.join(input_path, '*')
            # for images in glob.glob('*',root_dir = input_path):
            for images in glob.glob(pattern):
                images = os.path.basename(images)
                if images.endswith("_1.jpg"):
                    self.write_edge(input_path + images , output_path+ os.path.splitext(images)[0] +".png")
        
        gc.collect()
        command = "/root/miniconda3/envs/GsE/bin/python /root/autodl-tmp/GaussianVTON/ladi/src/inference.py --num_inference_steps 20 --dataset dresscode --dresscode_dataroot /root/autodl-tmp/GaussianVTON/ladi/input  --output_dir /root/autodl-tmp/GaussianVTON/ladi/results --test_order unpaired  --batch_size 3 --num_workers 2 --enable_xformers_memory_efficient_attention"
        # command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/src/inference.py --num_inference_steps 20 --dataset dresscode --dresscode_dataroot ./input  --output_dir ./results --test_order unpaired  --batch_size 3 --num_workers 2 --enable_xformers_memory_efficient_attention"
        subprocess.call(command, shell=True)

########## Refinement ##

        # dresscode = 'final'
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
            # cv2.imwrite(res_paths[iter].replace('results/unpaired', 'final').replace('_0.jpg', '_' + dresscode + '.jpg'), org_res)
            cv2.imwrite(res_paths[iter], org_res)  

        category_name = os.path.basename(os.path.dirname(prompt_path))
        final_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired'
        img_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/final'
        final_path = os.path.join(final_path, category_name)
        # print(final_path)
        # for filename in os.listdir(final_path):
        #     filepath = os.path.join(final_path, filename)
            
        #     img = Image.open(filepath)
            
        #     cropped_img = img.crop((0, 0, 512, 512))
            
        #     cropped_img.save(filepath)
        # final_tensor = self.load_images_as_tensor(final_path)
        for filename in os.listdir(final_path):
            
            filepath = os.path.join(final_path, filename)

            # print(filepath)

            # img = Image.open(filepath)
            bt, H, W, C = human.shape
            
            # img = img.crop((0, 0, 512, 512))
            target_size = (W, H)
            H = 512
            W = 292
            print(W)
            img = self.crop_and_resize_image(filepath, W, H)

            img_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/final'

            imgpath = os.path.join(img_path, filename)

            img.save(imgpath)

        SAM = LangSAM()

        if category_name == 'upper_body':
            text_prompt = "upper garment"
        elif category_name == 'lower_body':
            text_prompt = "pants"
        elif category_name == 'dresses':
            text_prompt = "dress"
        else:
            text_prompt = ""

        # text_prompt = "upper garment"

        unsam_images = glob.glob("/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/final/*.jpg")

        for unsam_image in unsam_images:
            image_pil = Image.open(unsam_image).convert("RGB")
            image = cv2.imread(unsam_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks, boxes, phrases, logits = SAM.predict(image_pil, text_prompt)

            for i, mask in enumerate(masks):
                # mask = mask.numpy
                mask = ~mask
                mask = mask + 255
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                mask_np = mask.numpy()
                mask = mask_np.astype(np.uint8)
                res = cv2.bitwise_and(image, mask)
                res[res == 0] = 255
                filename = os.path.basename(unsam_image)
                save_path = '/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/sam/' + filename
                # print("res shape:", res.shape)
                cv2.imwrite(save_path, res)

        sam_path = "/root/autodl-tmp/GaussianVTON/ladi/results/unpaired/sam"
        most_different_images = self.find_most_different_images(sam_path)
        print(most_different_images)

        # self.ladi_vton(most_different_images, prompt_path, img_path, category_name, H, W)

        # img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_tensor = self.load_images_tensor(img_path)

        # bt, H, W, C = human.shape
        # target_shape = (H, W)

        # resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='bilinear', align_corners=False)

        # print("img_tensor:", resized_tensor.shape)

        # print(final_tensor.shape)

        return {"edit_images": img_tensor}

if __name__ == "__main__": 
    import os