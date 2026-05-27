#v(1fr)

#heading(numbering: none)[Приложения]

#v(1fr)

#pagebreak()

#set text(size: 12pt)

#let application(name: none, content) = {
    set text(14pt)
    h(1fr)
    [
        #if name == none [
            ПРОДОЛЖЕНИЕ
        ]
        ПРИЛОЖЕНИЕ #content
    ]

    align(center)[#name]
}

#application(name: "Конфигурационный файл обучения StyleNet")[A]
```yaml
epochs: 100
use_amp: true

model:
  StyleNet:
    backbone: efficientnetv2_rw_s
    head:
      StyleNetHead:
        partial: true
        dropout: 0.3

loss:
  StyleNetLoss

optimizer:
  torch.optim.AdamW:
    lr: 0.0003
    weight_decay: 0.0001

train_loader:
  batch_size: 64
  shuffle: true
  num_workers: 12
  drop_last: false
  pin_memory: true
  dataset:
    BasicDataset:
      loader:
        FontDatasetLoader:
          root: dataset
          train: true

      input_shape: image
      target_shape:
        font_color: font_color
        stroke_color: stroke_color
        stroke_thickness: stroke_thickness

      transforms:
        - ConvertToLab:
            keys: [font_color, stroke_color]
            normalize: true
            source: hex
```
#pagebreak()
#application()[A]
```yaml
        - Albumentations:
            image_keys: [image]
            transforms:
              - albumentations.GaussianBlur:
                  blur_limit: [1, 3]
                  p: 0.2
              - albumentations.ImageCompression:
                  quality_range: [70, 100]
                  p: 0.3
              - albumentations.Downscale:
                  scale_range: [0.5, 0.9]
                  p: 0.3
              - albumentations.Perspective:
                  scale: [0.01, 0.03]
                  p: 0.15
              - albumentations.LongestMaxSize:
                  max_size: 224
              - albumentations.PadIfNeeded:
                  min_height: 224
                  min_width: 224
                  border_mode: 0

        - Normalize:
            keys: [image]
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]

        - ToTensor:
            keys: [image]
            scale: true
            dtype: float
val_loader:
  batch_size: 1
  shuffle: false
  num_workers: 2
  drop_last: false
  pin_memory: true
  dataset:
    BasicDataset:
      loader:
        FontDatasetLoader:
          root: dataset
          train: false
```
#pagebreak()
#application()[A]
```yaml
      input_shape: image
      target_shape:
        font_color: font_color
        stroke_color: stroke_color
        stroke_thickness: stroke_thickness

      transforms:
        - ConvertToLab:
            keys: [font_color, stroke_color]
            normalize: true
            source: hex

        - Albumentations:
            image_keys: [image]
            transforms:
              - albumentations.LongestMaxSize:
                  max_size: 224
              - albumentations.PadIfNeeded:
                  min_height: 224
                  min_width: 224
                  border_mode: 0

        - Normalize:
            keys: [image]
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]

        - ToTensor:
            keys: [image]
            scale: true
            dtype: float

callbacks:
  - MetricsCallback
  - ProgressCallback
  - CheckpointCallback:
      monitor: val_loss
      mode: min
      top_k: 3
```
#pagebreak()
#application(name: "Конфигурационный генерации датасета")[B]

```yaml
generator:
  FontDatasetGenerator:
    min_font_size: 24
    max_font_size: 48
    backgrounds_dir: assets/backgrounds

    min_margin: 20
    max_margin: 40
    rotation_offset: 30
    max_chars_per_line: 1000
    max_phrase_length: 3
    min_color_contrast: 75
    stroke_thickness: 8
    expansion: 64
    seed: 42

split: 0.9
output: dataset-lines/
```
