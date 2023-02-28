import weaviate
import numpy as np
from pathlib import Path
import requests
import time
import torch
import json
import os
import sys
import tqdm
import pickle
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))
from visual_clues.vlm_implementation import BlipItcVlmImplementation
from visual_clues.ontology_implementation import SingleOntologyImplementation
from lavis.models import load_model_and_preprocess

BLIP2_EMBEDDINGS_FOLDER = "blip2_embeddings"
DIR_PATH = os.path.dirname(__file__)
# from visual_clues.bboxes_implementation import DetectronBBInitter

URL_PREFIX = "http://74.82.29.209:9000"


class FeaturesMaker:
    def __init__(self):
        self.client = weaviate.Client("http://64.71.146.93:8080")
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.ontology_objects = SingleOntologyImplementation(
            'vg_objects', vlm_name="clip")
        self.ontology_places = SingleOntologyImplementation(
            'scenes', vlm_name="clip")
        self.ontology_attributes = SingleOntologyImplementation(
            'vg_attributes', vlm_name="clip")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=self.device)
        
    def create_class_object(self, class_name, description, properties_name):
        schema = self.client.schema.get()
        print(json.dumps(schema, indent=4))
        class_obj = {
            # <= Change to your class name - it will be your collection
            "class": class_name,
            "description": description,
            "vectorizer": "none",
            "properties": [
                {
                    "dataType": [
                        "string"
                    ],
                    "description": properties_name,
                    "name": properties_name
                }
            ]
        }
        self.client.schema.delete_class(class_name)
        self.client.schema.create_class(class_obj)

    def import_face_vectors(self, data, class_name, properties_name):

        for idx, vector in enumerate(tqdm.tqdm(data)):
            try:
                ontology_prompt_name = vector[0]
                embedding = vector[1].flatten(start_dim=1, end_dim=2)
                # print(embedding.shape)
                self.client.data_object.create(
                    data_object={properties_name: ontology_prompt_name},
                    class_name=class_name,
                    vector=embedding
                )
            except:
                print("Error")

    def get_text_feat(self, text):
        text = self.txt_processors["eval"](text)
        sample = {"text_input": [text]}
        features_text = self.model.extract_features(sample, mode="text")
        print(features_text.text_embeds_proj[:,0,:].t().shape)
        return features_text.text_embeds_proj[:,0,:].t()

    def create_ontology_embeddings(self, ontology_name):

        name_to_embedding = []
        DIR_PATH

        if ontology_name == "vg_objects":
            objects_text = self.ontology_objects.texts
            for object_text in objects_text:
                object_feat = self.get_text_feat(object_text)
                name_to_embedding.append([object_text, object_feat])
            vg_objects_path = os.path.join(os.path.join(DIR_PATH, BLIP2_EMBEDDINGS_FOLDER), "blip2_vg_objects.pkl")
            # self.save_data(vg_objects_path, name_to_embedding)

        elif ontology_name == "vg_attributes":
            attributes_text = self.ontology_attributes.texts
            for att_text in attributes_text:
                att_feat = self.get_text_feat(att_text)
                name_to_embedding.append([att_text, att_feat])
            vg_atts_path = os.path.join(os.path.join(DIR_PATH, BLIP2_EMBEDDINGS_FOLDER), "blip2_vg_atts.pkl")
            # self.save_data(vg_atts_path, name_to_embedding)

        elif ontology_name == "scenes":
            places_text = self.ontology_places.texts
            for place_text in places_text:
                place_feat = self.get_text_feat(place_text)
                name_to_embedding.append([place_text, place_feat])
            vg_places_path = os.path.join(os.path.join(DIR_PATH, BLIP2_EMBEDDINGS_FOLDER), "blip2_vg_places.pkl")
            # self.save_data(vg_places_path, name_to_embedding)

        else:
            print("Invalid ontology name.")
    
        return name_to_embedding

    def save_data(self, path, data):
        with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



def main():
    features_maker_instance = FeaturesMaker()

    # features_maker_instance.create_class_object(
    # class_name="blip2_scenes", description="embeddings of visual genome scenes", properties_name="scene_prompt")
    name_to_embedding = features_maker_instance.create_ontology_embeddings(
        ontology_name = "scenes")
    # features_maker_instance.import_face_vectors(
    #     data=name_to_embedding, class_name="blip2_scenes", properties_name="scene_prompt")

    # features_maker_instance.create_class_object(
        # class_name="blip2_objects", description="embeddings of visual genome objects", properties_name="object_prompt")
    name_to_embedding = features_maker_instance.create_ontology_embeddings(
        ontology_name = "vg_objects")
    # features_maker_instance.import_face_vectors(
        # data=name_to_embedding, class_name="blip2_objects", properties_name="object_prompt")
    
    # features_maker_instance.create_class_object(
    # class_name="blip2_attributes", description="embeddings of visual genome attributes", properties_name="att_prompt")
    name_to_embedding = features_maker_instance.create_ontology_embeddings(
        ontology_name = "vg_attributes")
    # features_maker_instance.import_face_vectors(
        # data=name_to_embedding, class_name="blip2_attributes", properties_name="att_prompt")



if __name__ == '__main__':
    main()
