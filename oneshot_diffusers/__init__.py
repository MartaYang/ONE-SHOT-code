"""ONE-SHOT's own diffusers override package.

It contains the two top-level classes that are actually modified by ONE-SHOT:

- ``WanOneshotTransformer3DModel``: the ONESHOT rewritten version, incompatible with upstream diffusers
- ``WanOneshotPipeline``: the ONESHOT inference pipeline, which does not exist upstream

It also includes the sibling utility ``oneshot_util.py`` used by ``transformer_wan_oneshot.py``.

Other diffusers symbols, such as ``AutoencoderKLWan`` / ``FlowMatchEulerDiscreteScheduler``,
should be imported directly from the upstream ``diffusers`` package. See the repository
``requirements.txt`` for the required version.

The ``model_index.json`` in the model directory points both the pipeline and transformer to the
``oneshot_diffusers`` namespace, e.g., ``["oneshot_diffusers", "WanOneshotXxx"]``.
During ``from_pretrained``, diffusers will call ``importlib.import_module("oneshot_diffusers")``
and then use ``getattr`` to retrieve the class, so no monkey-patching is needed.
"""
from .transformer_wan_oneshot import WanOneshotTransformer3DModel
from .pipeline_wan_oneshot import WanOneshotPipeline

# Re-export so diffusers' pipeline_loading_utils.get_class_obj_and_candidates
# can find ``ModelMixin`` as an attribute of the ``oneshot_diffusers`` library
# when ``model_index.json`` references our package. Without this re-export,
# diffusers' subclass-of-loadable-base check fails with
# "cannot be loaded as it does not seem to have any of the loading methods".
from diffusers import ModelMixin  # noqa: F401

__all__ = ["WanOneshotTransformer3DModel", "WanOneshotPipeline", "ModelMixin"]