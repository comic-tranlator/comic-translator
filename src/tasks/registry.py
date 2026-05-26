from .style_extraction import StyleExtractor

REGISTRY = {cls.__name__: cls for cls in (StyleExtractor,)}
