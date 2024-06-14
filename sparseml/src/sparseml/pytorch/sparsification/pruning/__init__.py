# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pruning modifiers and utilities to support their creation
"""

# flake8: noqa


from .mask_creator import *
from .mask_params import *
from .modifier_as import *
from .modifier_pruning_acdc import *
from .modifier_pruning_base import *
from .modifier_pruning_constant import *
from .modifier_pruning_layer import *
from .modifier_pruning_magnitude import *
from .modifier_pruning_mfac import *
from .modifier_pruning_movement import *
from .modifier_pruning_structured import *
from .scorer import *
from .permutations import *

from .modifier_pruning_obs import * # 8-block+unstructured

from .modifier_pruning_obs_28 import * # one-shot 2:8
from .modifier_pruning_obs_216 import * # one-shot 2:16

from .modifier_pruning_obs_28_gradual import *
from .modifier_pruning_obs_216_gradual import *

from .modifier_pruning_28_pairwise_obs import *
from .modifier_pruning_216_pairwise_obs import *

from .modifier_pruning_obs_28v64 import *
from .modifier_pruning_obs_28v128 import *
from .modifier_pruning_obs_216v64 import *
from .modifier_pruning_obs_216v128 import *

from .modifier_pruning_obs_28v64_gradual import *
from .modifier_pruning_obs_216v64_gradual import *
from .modifier_pruning_obs_28v128_gradual import *
from .modifier_pruning_obs_216v128_gradual import *

from .modifier_pruning_28_pairwise_obs_v64 import *
from .modifier_pruning_216_pairwise_obs_v64 import *
from .modifier_pruning_28_pairwise_obs_v128 import *
from .modifier_pruning_216_pairwise_obs_v128 import *

from .modifier_pruning_hinm import *
from .modifier_pruning_28_pairwise_obs_v64_perm import *
from .modifier_pruning_28_pairwise_obs_v128_perm import *
from .modifier_pruning_216_pairwise_obs_v64_perm import *
from .modifier_pruning_216_pairwise_obs_v128_perm import *
#from .gpu_profile import *
