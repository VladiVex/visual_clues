import numpy as np
from database.arangodb import DatabaseConnector
from config.config import NEBULA_CONF
import cv2
from pathlib import Path
import csv
import requests

from movie.movie_db import MOVIE_DB
import tqdm
from PIL import Image

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from visual_clues.ontology_implementation import SingleOntologyImplementation
from visual_clues.blip import BLIP_Captioner
from visual_clues.yolov7_implementation import YoloTrackerModel
# from visual_clues.bboxes_implementation import DetectronBBInitter

URL_PREFIX = "http://74.82.29.209:9000"

class TokensPipeline:
    def __init__(self):
        self.config_db = NEBULA_CONF()
        self.db_host = self.config_db.get_database_host()
        self.database = self.config_db.get_playground_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.nre = MOVIE_DB()
        self.db = self.nre.db
        self.blip_captioner = BLIP_Captioner()
        self.ontology_objects = SingleOntologyImplementation('vg_objects', vlm_name="blip_itc")
        self.ontology_places = SingleOntologyImplementation('scenes', vlm_name="blip_itc")
        self.ontology_attributes = SingleOntologyImplementation('vg_attributes', vlm_name="blip_itc")
        self.yolo_detector = YoloTrackerModel()
        # self.det_proposal = DetectronBBInitter()


    def load_img_url(self, img_url : str, pil_type=False):
        # Load PIL Image
        if pil_type:
            resp = requests.get(img_url, stream=True).raw
            image = Image.open(resp).convert('RGB')
        # Load OpenCV Image
        else:
            resp = requests.get(img_url, stream=True).raw
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    
    def compute_scores(self, ontology, img, top_n = 10):
        """
        Returns top n ontology list and its corresponding scores sorted in reverse order.
        """
        ontology_scores = ontology.compute_scores(img)
        sorted_scores = sorted(ontology_scores, key=lambda x: x[1], reverse=True)
        sorted_scores_ = [(score[0], str(score[1])) for score in sorted_scores]
        scores = sorted_scores_[:top_n]
        return scores

    def insert_json_to_db(self, combined_json, db_name="nebula_playground"):
        """
        Inserts a JSON with global & local tokens to the database.
        """

        self.nre.change_db("prodemo")
        self.db = self.nre.db

        query = 'UPSERT { movie_id: @movie_id, frame_num: @frame_num } INSERT  \
                { movie_id: @movie_id, frame_num: @frame_num, roi: @roi, url: @url, global_objects: @global_objects, global_caption: @global_caption,\
                            global_scenes: @global_scenes, source: @source\
                        } UPDATE {movie_id: @movie_id, frame_num: @frame_num, roi: @roi, url: @url, global_objects: @global_objects, global_caption: @global_caption,\
                            global_scenes: @global_scenes, \
                            source: @source} IN s4_visual_clues'

        self.db.aql.execute(query, bind_vars=combined_json)
        print("Successfully inserted to database.")
        return
        

    def create_json_global_tokens(self, movie_id, mdf, global_objects,
                                    global_caption, global_scenes, img_url, source):
        """
        Returns a JSON filled with global tokens.
        """
        json_global_tokens = {
                    "movie_id": movie_id,
                    "frame_num": mdf,
                    "global_objects": { 'blip' : global_objects },
                    "global_caption": { 'blip' : global_caption },
                    "global_scenes":  { 'blip' : global_scenes  },
                    "url": img_url,
                    "source": source
        }
        return json_global_tokens

    def create_json_local_tokens(self, movie_id, mdf, local_dict, img_url, source="None"):
        
        json_local_tokens = {
            "movie_id": movie_id,
            "frame_num": mdf,
            "roi": local_dict,
            "url": img_url,
            "source": source
        }
        return json_local_tokens

    def create_combined_json(self, glob_tkns_json, loc_tkns_json):
        """
        Returns a JSON filled with global and local tokens.
        """
        combined_json = glob_tkns_json | loc_tkns_json
        return combined_json
    

    def create_local_tokens(self, img_url, movie_id, mdf):
        """
        Returns a JSON with local tokens for an image url.
        """

        # cv_img = self.load_img_url(img_url, pil_type=False)
        # bbox_proposals = self.det_proposal.compute_bbox_proposals(cv_img)
        # bb_rescale_ratio = [inp/out for out,inp in zip(bbox_proposals['image_size'][:2], cv_img.shape)]
        # pil_img = self.load_img_url(img_url, pil_type=True)

        # bbox_propsals_objs = []
        # bbox_propsals_attrs = []

        local_dict = []

        cv_img = self.load_img_url(img_url, pil_type=False)
        yolo_output = self.yolo_detector.forward(cv_img)

        for idx, output in enumerate(yolo_output):
            cur_obj, cur_bbox, cur_conf = output['detection_classes'], output['detections_boxes'], output['detection_scores']

            local_dict.append({
                'roi_id': str(idx),
                'bbox': cur_bbox,
                'bbox_object': cur_obj,
                'bbox_confidence': cur_conf,
                'bbox_source': 'yolov7'
            })

        # for idx, bbox in enumerate(bbox_proposals['meta_data_det']):
        #     scaled_bbox = [bbox[0]*bb_rescale_ratio[1], bbox[1]*bb_rescale_ratio[0],
        #                     bbox[2]*bb_rescale_ratio[1], bbox[3]*bb_rescale_ratio[0]]

        #     # Objects on ROI proposals.
        #     scores_obj = self.ontology_objects.compute_scores_with_bboxes(pil_img, scaled_bbox)  
        #     scores_obj_sorted = sorted(scores_obj, key=lambda x: x[1], reverse=True)
        #     scores_obj_sorted_ = [(score[0], str(score[1])) for score in scores_obj_sorted]

        #     bbox_propsals_objs.append({str(scaled_bbox) : scores_obj_sorted_})

        #     # Attributes on ROI proposals.
        #     scores_obj = self.ontology_attributes.compute_scores_with_bboxes(pil_img, scaled_bbox)  
        #     scores_attr_sorted = sorted(scores_obj, key=lambda x: x[1], reverse=True)
        #     scores_attr_sorted_ = [(score[0], str(score[1])) for score in scores_attr_sorted]
            
        #     bbox_propsals_attrs.append({str(scaled_bbox) : scores_attr_sorted_})

        #     img = self.load_img_url(img_url, pil_type=True)
        #     cropped_image = img.crop((scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3]))
        #     processed_frame = self.blip_captioner.process_frame(cropped_image)
        #     bbox_caption = self.blip_captioner.generate_caption(processed_frame)

        #     local_dict.append({
        #         'roi_id': str(idx + len(yolo_output)),
        #         'bbox': str(bbox),
        #         'bbox_source': 'rpn',
        #         'local_captions': {'blip': bbox_caption},
        #         'local_objects': {'blip' : scores_obj_sorted_},
        #         'local_attributes': {'blip':scores_attr_sorted_},
        #     })
        
        
        json_local_tokens = self.create_json_local_tokens(movie_id, mdf, local_dict=local_dict,
                                                            img_url=img_url, source="None")
                                                            

        return json_local_tokens


    def create_global_tokens(self, img_url, movie_id, frame_num):
        """
        Returns a JSON with global tokens for an image url.
        """
        pil_img = self.load_img_url(img_url, pil_type=True)

        scores_objects = self.compute_scores(self.ontology_objects, pil_img, top_n = 10)
        scores_places = self.compute_scores(self.ontology_places, pil_img, top_n = 10)

        pil_img = self.load_img_url(img_url, pil_type=True)
        processed_frame = self.blip_captioner.process_frame(pil_img)
        caption = self.blip_captioner.generate_caption(processed_frame)

        json_global_tokens = self.create_json_global_tokens(movie_id = movie_id, mdf=frame_num, global_objects=scores_objects,
                                                        global_caption=caption,
                                                         global_scenes=scores_places, img_url=img_url, source="None")

        return json_global_tokens

    def get_mdf_urls_from_db(self, movie_id):
        print("Trying to get movie {} from db:".format(movie_id))
        print(self.nre.db)
        data = self.nre.get_movie(movie_id=movie_id)
        urls = []
        if not data:
            print("{} not found in database. ".format(movie_id))
            return False
        if 'mdfs_path' not in data:
            print("MDFs cannot be found in {}".format(movie_id))
            return False
        for mdf_path in data['mdfs_path']:
            url = os.path.join(URL_PREFIX, mdf_path[1:])
            urls.append(url)
        return urls
    
    def check_image_url(self, img_url):
        resp = requests.get(img_url, stream=True).raw
        if not resp.reason == 'OK':
            print("Image URL: {} couldn't be loaded succesfully.".format(img_url))
            return False
        return True
        
    def run_visual_clues_pipeline(self, movie_id):
        image_urls = self.get_mdf_urls_from_db(movie_id)
        length_urls = len(image_urls)
        if length_urls == 0:
            return False, None
        for idx, img_url in enumerate(image_urls):
            print("Working on current image url: {}".format(img_url))
            img_url_is_valid = self.check_image_url(img_url)
            if img_url_is_valid:
                if len(image_urls) == 1:
                    cur_frame_num = 0
                else:
                    cur_frame_num = int(img_url.split("/")[-1].split(".jpg")[0].replace("frame",""))
                glob_tkns_json = self.create_global_tokens(img_url, movie_id, cur_frame_num)
                loc_tkns_json = self.create_local_tokens(img_url, movie_id, cur_frame_num)
                combined_json = self.create_combined_json(glob_tkns_json, loc_tkns_json)
                self.insert_json_to_db(combined_json)
                counter = idx + 1
                print("Finished with {}/{}".format(counter, length_urls))
            else:
                counter = idx + 1
                print("Finished with {}/{}".format(counter, length_urls))
                print("Skipping the invalid image URL: {}".format(img_url))
        return True, None


def main():
    tokens_pipeline = TokensPipeline()
    img_url = 'https://cs.stanford.edu/people/rak248/VG_100K/2316634.jpg'
    movie_id, cur_frame_num = "test", "1"
    glob_tkns_json = tokens_pipeline.create_global_tokens(img_url, movie_id, cur_frame_num)
    loc_tkns_json = tokens_pipeline.create_local_tokens(img_url, movie_id, cur_frame_num)
    combined_json = tokens_pipeline.create_combined_json(glob_tkns_json, loc_tkns_json)
    tokens_pipeline.insert_json_to_db(combined_json)
    print("Done")



if __name__ == '__main__':
    main()
