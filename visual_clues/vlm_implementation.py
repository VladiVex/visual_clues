from visual_clues.vlm_interface import VlmInterface
from visual_clues.utils.config import config
import typing
from PIL import Image
import requests
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval
from visual_clues.models.blip_itm import blip_itm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os.path
import wget
from pathlib import Path
from functools import lru_cache, wraps
from time import sleep
import io
import torch.nn.functional as F
import time
import pickle
from torch.hub import set_dir
BLIP2_EMBEDDINGS_FOLDER = "blip2_embeddings"
DIR_PATH = os.path.dirname(__file__)
vg_objects_path = os.path.join(os.path.join(DIR_PATH, BLIP2_EMBEDDINGS_FOLDER), "blip2_vg_objects.pkl")
vg_atts_path = os.path.join(os.path.join(DIR_PATH, BLIP2_EMBEDDINGS_FOLDER), "blip2_vg_atts.pkl")
vg_places_path = os.path.join(os.path.join(DIR_PATH, BLIP2_EMBEDDINGS_FOLDER), "blip2_vg_places.pkl")
# from nebula3_experts_vg.vg.visual_grounding_inference import OfaMultiModalVisualGrounding
# from nebula3_videoprocessing.videoprocessing.owl_vit_impl import OwlVitImplementation

def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

class VlmBaseImplementation(VlmInterface):

    def compute_similarity_url(self, url: str, text: list[str]):
        image = self.load_image_url(url)
        return self.compute_similarity(image, text)

class VlmChunker(VlmBaseImplementation):
    def __init__(self, vlm: VlmInterface, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.vlm = vlm
    def load_image_url(self,url):
        return self.vlm.load_image_url(url)
    
    def compute_similarity(self, image: Image, text: list[str]) -> list[float]:
        results = []
        chunked_texts=[text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        for chunk in chunked_texts:
            results.extend(self.vlm.compute_similarity(image,chunk))
        return results  

class VisualGroundingToVlmAdapter(VlmBaseImplementation):

    def __init__(self): # vg : VgInterface
        self.vg = OwlVitImplementation()
    
    def compute_similarity(self, image, text):
        scored_bboxes = self.vg.ground_objects_batch(image, text)
        if not scored_bboxes:
            return []

        scores = []
        for idx in range(len(scored_bboxes) - 1):
            if not scored_bboxes[idx]:
                scores.append(0.0)
                continue
            else:
                max_conf = sorted(scored_bboxes[idx],key=lambda x: x[1], reverse=True)[0][1]#max([scored_bboxes[idx]], key=lambda item:item[1])[1]
                scores.append(max_conf)

        return scores

class ClipVlmImplementation(VlmBaseImplementation):

    def __init__(self, init_with_cpu=False):

        if init_with_cpu:
            print("Initializing model on CPU")
            self.device = torch.device('cpu')
        else:
            print("Initializing model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CLIPModel.from_pretrained(config["clip_checkpoints"]).to(device=self.device)
        self.processor = CLIPProcessor.from_pretrained(config["clip_checkpoints"])


    def load_image_url(self, url: str):
        return Image.open(requests.get(url, stream=True).raw)  

    def compute_similarity(self, image : Image, text : list[str]):

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(device=self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeds_dotproduct = (outputs.image_embeds.expand_as(outputs.text_embeds) * outputs.text_embeds).sum(dim=1)
        return embeds_dotproduct.cpu().detach().numpy()

class BlipItmVlmImplementation(VlmBaseImplementation):
    def __init__(self, init_with_cpu = False):

        if init_with_cpu:
            print("Initializing model on CPU")
            self.device = torch.device('cpu')
        else:
            print("Initializing model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if not os.path.isfile(config['blip_model_url_large']):
        #     print("Blip Checkpoints not found locally, Downloading in progres...")
        #     dirs_path = "/" + '/'.join(config['blip_model_url_large'].split("/")[1:-1]) + "/"
        #     Path(dirs_path).mkdir(parents=True, exist_ok=True)
        #     wget.download(config['blip_model_url_large_url'], config['blip_model_url_large'])
        #     print("Successfully downloaded BLIP checkpoints.")

        # model = blip_itm(pretrained=config['blip_model_url_large'], image_size=config['blip_image_size'], vit=config['blip_vit_large'])
        # model.eval()
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
        self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device=self.device)
    
    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
        return image

    def load_image(self, image: Image): 
        transform = transforms.Compose([
            transforms.Resize((config['blip_image_size'], config['blip_image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(image).unsqueeze(0).to(self.device)   
        return image

    def compute_similarity(self, image: Image, text: list[str]):
        
        # image = self.load_image(image)

        # with torch.no_grad():
        #     itm_output = self.model(image, text, match_head='itm')
        
        # Change from softmax to dotproduct
        inputs = processor(raw_image, text, return_tensors="pt", padding=True).to("cuda")

        itm_scores = model(**inputs)[0]
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        itm_scores = itm_score.cpu().detach().numpy()

        return itm_scores

class BlipItcVlmImplementation(VlmBaseImplementation):
    def __init__(self, init_with_cpu = False):
        if init_with_cpu:
            print("Warning: Initializing BLIP_ITC model on CPU")
            self.device = 'cpu'
        else:
            print("Warning: Initializing BLIP_ITC model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # if not os.path.isfile(config['blip_model_url_large']):
        #     print("Blip Checkpoints not found locally, Downloading in progres...")
        #     dirs_path = "/" + '/'.join(config['blip_model_url_large'].split("/")[1:-1]) + "/"
        #     Path(dirs_path).mkdir(parents=True, exist_ok=True)
        #     wget.download(config['blip_model_url_large_url'], dirs_path)
        #     print("Successfully downloaded BLIP checkpoints.")

        # self.half = True if self.device != 'cpu' else False
        # model = blip_itm(pretrained=config['blip_model_url_large'], image_size=config['blip_image_size'], vit=config['blip_vit_large'])
        # model.eval()
        # self.model = model.to(device=self.device)
        # self.model = model.half() if self.half else self.model
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
        self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device=self.device)

    
    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB') 
        return image

    def load_image(self, image):   
        
        transform = transforms.Compose([
            transforms.Resize((config['blip_image_size'], config['blip_image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        if self.half:
            image = transform(image).half().unsqueeze(0).to(self.device)  
        else:
            image = transform(image).unsqueeze(0).to(self.device)
        return image

    def compute_similarity(self, image : Image, text : list[str]):
        # image = self.load_image(image)
        with torch.no_grad():
            inputs = self.processor(image, text, return_tensors="pt", padding=True).to("cuda")
            cosine_score = self.model(**inputs, use_itm_head=False)[0]
        itc_scores = cosine_score.cpu().detach().numpy()[0]

        return itc_scores
            # itc_output = self.model(image, text, match_head='itc')
        # Check if its dotproduct
        # itc_scores = itc_output.cpu().detach().numpy()[0]
        # return itc_scores

    @lru_cache()
    def get_cached_image_feat(self, img_byte_arr):
        img = img_byte_arr.getvalue()
        image = Image.open(io.BytesIO(img))
        image = self.load_image(image)
        image_embeds = self.model.visual_encoder(image) 
        image_feat = F.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1)

        return image_feat
    
    @lru_cache()
    def get_cached_text_feat(self, txt: tuple):
        txt = list(txt)
        text = self.model.tokenizer(txt, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(self.device) 
        text_output = self.model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
        text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)                                    
        return text_feat

    def compute_cached_similarity(self, image: Image, text: list[str]):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        with torch.no_grad():
            image_feat = self.get_cached_image_feat(img_byte_arr)
            text_feat = self.get_cached_text_feat(tuple(text))
            sim = image_feat @ text_feat.t()
        sim = sim.cpu().detach().numpy()[0]
        return sim

    
    # def bbox_xywh_to_xyxy(self, xywh):
    #     w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
    #     return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    
    def crop_image(self, image : Image, bbox: list[float]):
        # xmin, ymin, xmax, ymax = self.bbox_xywh_to_xyxy((bbox[0],bbox[1],bbox[2],bbox[3]))
        cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # cropped_image.save("/notebooks/test123.jpg")
        return cropped_image
    
    def compute_similarity_on_bboxes(self, image : Image, text : list[str], bbox : list[float]):

        cropped_image = self.crop_image(image, bbox)
        return self.compute_similarity(cropped_image, text)


class Blip_2_ItcVlmImplementation(VlmBaseImplementation):
    def __init__(self, init_with_cpu = False):
        from lavis.models import load_model_and_preprocess # because of packages incompatibility with transformers package
        if init_with_cpu:
            print("Warning: Initializing BLIP2_ITC model on CPU")
            self.device = 'cpu'
        else:
            print("Warning: Initializing BLIP2_ITC model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_dir('/inputs')

        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_feature_extractor", "pretrain", device=self.device, is_eval=True)
        self.model = self.model.float()
        blip2_names_to_obj_embeddings = self.read_data(vg_objects_path)
        self.blip2_names_to_obj_embeddings = {blip2_names_to_obj_embeddings[i][0]: blip2_names_to_obj_embeddings[i][1] for i in range(0, len(blip2_names_to_obj_embeddings))}

        blip2_names_to_att_embeddings = self.read_data(vg_atts_path)
        self.blip2_names_to_att_embeddings = {blip2_names_to_att_embeddings[i][0]: blip2_names_to_att_embeddings[i][1] for i in range(0, len(blip2_names_to_att_embeddings))}

        blip2_names_to_places_embeddings = self.read_data(vg_places_path)
        self.blip2_names_to_places_embeddings = {blip2_names_to_places_embeddings[i][0]: blip2_names_to_places_embeddings[i][1] for i in range(0, len(blip2_names_to_places_embeddings))}
        
        self.ontology_names_to_all_embeddings = {}
        self.ontology_names_to_all_embeddings.update(self.blip2_names_to_obj_embeddings)
        self.ontology_names_to_all_embeddings.update(self.blip2_names_to_att_embeddings)
        self.ontology_names_to_all_embeddings.update(self.blip2_names_to_places_embeddings)


    def read_data(self, path, mode="pickle"):
        if mode == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
                return data

    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB') 
        return image

    def load_image(self, image : Image):   
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        image = self.model.extract_features({"image": image}, mode="image")
        image = image.image_embeds_proj
        return image

    def compute_similarity(self, image : Image, text : list[str]):
        itc_scores = []
        with torch.no_grad():
            image_feat = self.load_image(image)
            for cur_text in text:
                if cur_text in self.ontology_names_to_all_embeddings:
                    text_feat = self.ontology_names_to_all_embeddings[cur_text][:,0,:].t()
                else:
                    text_feat = self.get_cached_text_feat(cur_text)
                itc_scores.append(float((image_feat.cuda() @ text_feat.cuda()).max().cpu()))

        return itc_scores

    @lru_cache()
    def get_cached_image_feat(self, img_byte_arr):
        img = img_byte_arr.getvalue()
        image = Image.open(io.BytesIO(img))
        image_feat = self.load_image(image)
        return image_feat
    
    @lru_cache()
    def get_cached_text_feat(self, txt: list): 
        text = self.text_processors["eval"](txt)
        text_feat = self.model.extract_features({"text_input": [text]}, mode="text")     
        text_feat = text_feat.text_embeds_proj[:,0,:].t()          
        return text_feat

    def compute_cached_similarity(self, image: Image, text: list[str]):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        t1 = time.time()
        itc_scores = []
        with torch.no_grad():
            image_feat = self.get_cached_image_feat(img_byte_arr)
            for cur_text in text:
                if cur_text in self.ontology_names_to_all_embeddings:
                    text_feat = self.ontology_names_to_all_embeddings[cur_text][:,0,:].t()
                else:
                    text_feat = self.get_cached_text_feat(cur_text)
                itc_scores.append(float((image_feat @ text_feat.cuda()).max().cpu())) 
        return itc_scores

    def crop_image(self, image : Image, bbox: list[float]):
        # xmin, ymin, xmax, ymax = self.bbox_xywh_to_xyxy((bbox[0],bbox[1],bbox[2],bbox[3]))
        cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # cropped_image.save("/notebooks/test123.jpg")
        return cropped_image
    
    def compute_similarity_on_bboxes(self, image : Image, text : list[str], bbox : list[float]):

        cropped_image = self.crop_image(image, bbox)
        return self.compute_similarity(cropped_image, text)
    
class VisualGroundingVlmImplementation(VlmInterface):
        def __init__(self):
            self.vg_engine = OfaMultiModalVisualGrounding()
        
        def load_image(self):
            pass

        def compute_similarity(self, image : Image, text : list[str]):
            time_measure = False
            if time_measure:
                import time
                since = time.time()

            bb, _, lprob = self.vg_engine.find_visual_grounding(image, text)

            if time_measure:
                time_elapsed = time.time() - since
                print('OFA VG time {:.3f}s'.format(time_elapsed))

            lprob = lprob.sum()
            debug = False
            if debug:
                plot_vg_over_image(bb, image, caption=text, lprob=lprob)

            return bb, lprob.cpu().numpy()


def main():
    ### CLIP USAGE EXAMPLE ###
    # clip_vlm = ClipVlmImplementation()

    # url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    # text =['a woman sitting on the beach with a dog', 'a man standing on the beach with a cat']
    # similarity = clip_vlm.compute_similarity_url(url, text)
    # print(f"CLIP outputs: {similarity}")

    # ##################################

    # ### BLIP ITM USAGE EXAMPLE ###

    # blip_vlm = BlipItmVlmImplementation()
    # text = ['a woman sitting on the beach with a dog']
    # similarity = blip_vlm.compute_similarity_url(url, text)
    # itm_score = similarity
    # print(f"BLIP_ITM outputs:")
    # print('The image and text is matched with a probability of %.4f'%itm_score)

    ## BLIP ITC USAGE EXAMPLE ###
    blip_vlm = BlipItcVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    similarity = blip_vlm.compute_similarity_url(url, text)
    itc_score = similarity
    print(f"BLIP_ITC outputs:")
    print('The image and text is matched with a probability of %.4f'%itc_score)

    text = ['a woman sitting on the beach with a dog']
    similarity = blip_vlm.compute_similarity_url(url, text)
    itc_score = similarity
    print(f"BLIP_ITC outputs:")
    print('The image and text is matched with a probability of %.4f'%itc_score)




if __name__ == "__main__":
    # main()
    pass
