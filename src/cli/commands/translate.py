from pathlib import Path

import click
import yaml

from src.pipeline import PipelineConfig, TranslationPipeline
from src.util.file import load_file
from src.util.path import CONFIG_DIR


@click.command(name="translate")
@click.argument("file", required=True, type=click.Path(path_type=Path))
@click.option("-s", "--source", default="english", type=str)
@click.option("-t", "--target", default="russian", type=str)
@click.option(
    "-o",
    "--output",
    default=Path("output.pdf"),
    type=click.Path(path_type=Path),
)
def translate_cmd(file: Path, source: str, target: str, output: Path):
    click.echo("Loading pages...")
    pages = load_file(file)[:5]
    click.echo(f"Loaded {len(pages)} pages")

    config_raw = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    config = PipelineConfig(**config_raw)

    pipeline = TranslationPipeline(source, target, config)

    first_page, *rest = list([page.convert("RGB") for page in pipeline(pages)])

    first_page.save(
        output,
        save_all=True,
        append_images=rest,
    )
