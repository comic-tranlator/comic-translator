type DataShape[T] = T | list[T] | dict[str, T]


def shape_data[T](sample: dict[str, T], shape: DataShape[str]) -> DataShape[T]:
    if isinstance(shape, str):
        return sample[shape]
    if isinstance(shape, tuple):
        return [sample[field] for field in shape]
    if isinstance(shape, dict):
        return {alias: sample[field] for alias, field in shape.items()}
    raise ValueError(f"Unsupported DataShape: {shape!r}")


