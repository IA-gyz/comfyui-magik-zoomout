from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .magik_zoom_out import MagikZoomOut


class MagikZoomOutExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MagikZoomOut]


async def comfy_entrypoint() -> MagikZoomOutExtension:
    return MagikZoomOutExtension()
