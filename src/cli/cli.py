import click

from .commands import (
    collect_backgrounds_cmd,
    generate_dataset_cmd,
    train_cmd,
    translate_cmd,
)


@click.group()
@click.version_option("1.0.0")
def cli():
    pass


cli.add_command(collect_backgrounds_cmd)
cli.add_command(generate_dataset_cmd)
cli.add_command(train_cmd)
cli.add_command(translate_cmd)
