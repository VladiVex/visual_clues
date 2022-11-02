from dataclasses import dataclass
from .constants import OUTPUT_JSON, TYPE_MOVIE

@dataclass
class ExpertParam:
    id: str
    scene_element: int = None
    local: bool = False
    extra_params: dict = None
    output: str = OUTPUT_JSON
    type: str = TYPE_MOVIE
    img_url: str = None
    output_file: str = None


@dataclass
class TokenRecord:
    movie_id: str
    scene_element: int = 0
    scene: int = 0
    expert: str = None
    bbox: list = None
    label: str = None
    meta_label: dict = None
    re_id: int = 0

@dataclass
class ImageRecord:
    image_id: str
    expert: str = None
    bbox: list = None
    label: str = None
    meta_label: dict = None
    re_id: int = 0

@dataclass
class ImageToken:
    id: str # "22094333" / "Movies/308719/1/0"
    type: str # 'image" / "frame"
    expert: str = None # "places"
    expert_meta: dict = None # { model: 'blip', type: 'retrieval' } / { model: '365', type: 'classification' }
    bbox: list = None
    label: str = None
    label_meta: dict = None
    score: float = None
    re_id: int = 0
