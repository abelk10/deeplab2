"""File containing the meta info of Graph dataset."""

import copy
from typing import Sequence, Any

_GRAPH_META = [{
    "color": [255, 255, 255],
    "isthing": 0,
    "name": "background",
    "id": 1
},{
    "color": [120, 120, 120],
    "isthing": 1,
    "name": "node",
    "id": 1
}, {
    "color": [180, 120, 120],
    "isthing": 2,
    "name": "edge",
    "id": 2
}]


def get_graph_meta() -> Sequence[Any]:
  return copy.deepcopy(_GRAPH_META)


def get_graph_class_has_instances_list() -> Sequence[int]:
  return tuple([x["id"] for x in _GRAPH_META if x["isthing"] == 1])


def get_id_mapping_inverse() -> Sequence[int]:
  id_mapping_inverse = (255,) + tuple(range(150))
  return id_mapping_inverse

