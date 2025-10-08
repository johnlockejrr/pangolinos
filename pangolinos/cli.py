#
# Copyright 2025 johnlockejrr / Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
pangolinos.cli
~~~~~~~~~~~~~

Command line driver for rendering text.
"""
import click
import logging

from pathlib import Path
from rich.progress import Progress
from multiprocessing import Pool
from functools import partial
from itertools import zip_longest

from pangolinos.render import _markup_colors
from pangolinos.polygonize import polygonize_document

from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

logging.captureWarnings(True)
logger = logging.getLogger('pangolinos')


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        try:
            for entry in manifest.readlines():
                im_p = Path(entry.rstrip('\r\n'))
                if im_p.is_file():
                    images.append(im_p)
                else:
                    logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
        except UnicodeDecodeError:
            raise click.BadOptionUsage(param,
                                       f'File {manifest.name} is not a text file. Please '
                                       'ensure that the argument to `-W` is a manifest '
                                       'file containing paths to image file (one per '
                                       'line).',
                                       ctx=ctx)
    return images


@click.group(chain=False)
@click.version_option()
@click.option('--workers', show_default=True, default=1, type=click.IntRange(1), help='Number of worker processes.')
def cli(workers):
    """
    Base command for the text renderer
    """
    ctx = click.get_current_context()
    ctx.meta['workers'] = workers


def _render_doc(doc, output_dir, paper_size, margins, font, language,
                base_dir, enable_markup, random_markup,
                random_markup_probability, skip_unrenderable, line_spacing,
                padding_all, padding_horizontal, padding_vertical,
                padding_left, padding_right, padding_top, padding_bottom, padding_baseline, 
                use_polygons, use_fine_contours):
    from pangolinos.render import render_text

    with open(doc, 'r') as fp:
        render_text(text=fp.read(),
                    output_base_path=output_dir / doc.name,
                    paper_size=paper_size,
                    margins=margins,
                    font=font,
                    language=language,
                    base_dir=base_dir,
                    enable_markup=enable_markup,
                    random_markup=random_markup,
                    random_markup_probability=random_markup_probability,
                    raise_unrenderable=not skip_unrenderable,
                    line_spacing=line_spacing,
                    padding_all=padding_all,
                    padding_horizontal=padding_horizontal,
                    padding_vertical=padding_vertical,
                    padding_left=padding_left,
                    padding_right=padding_right,
                    padding_top=padding_top,
                    padding_bottom=padding_bottom,
                    padding_baseline=padding_baseline,
                    use_polygons=use_polygons,
                    use_fine_contours=use_fine_contours)


@cli.command('render')
@click.pass_context
@click.option('-p', '--paper-size', default=(210, 297), show_default=True,
              type=(int, int),
              help='Paper size `(width, height)` in mm.')
@click.option('-m', '--margins', default=(25, 30, 25, 25), show_default=True,
              type=(int, int, int, int),
              help='Page margins `(top, bottom, left, right)` in mm.')
@click.option('-f', '--font', default='Serif Normal 10', show_default=True,
              help='Font specification to render the text in.')
@click.option('-l', '--language', default=None,
              help='Language in country code-language format to set for '
              'language-specific rendering. If none is set, the system '
              'default will be used.')
@click.option('-b', '--base-dir', default=None, type=click.Choice(['L', 'R']),
              help='Base direction for Unicode BiDi algorithm.')
@click.option('-O', '--output-dir',
              type=click.Path(exists=False,
                              dir_okay=True,
                              file_okay=False,
                              writable=True,
                              path_type=Path),
              show_default=True,
              default=Path('.'),
              help='Base output path to place PDF and XML outputs into.')
@click.option('--markup/--no-markup',
              default=False,
              help='Switch for Pango markup parsing in input texts.')
@click.option('--random-markup', default=['style_italic', 'weight_bold',
                                          'underline_single',
                                          'underline_double',
                                          'overline_single', 'shift_subscript',
                                          'shift_superscript', 'strikethrough_true'],
              type=click.Choice(['style_oblique', 'style_italic',
                                 'weight_ultralight', 'weight_bold',
                                 'weight_ultrabold', 'weight_heavy',
                                 'variant_smallcaps', 'underline_single',
                                 'underline_double', 'underline_low',
                                 'underline_error', 'overline_single',
                                 'shift_subscript', 'shift_superscript',
                                 'strikethrough_true', 'foreground_random'] + [f'foreground_{x}' for x in _markup_colors]),
              multiple=True, show_default=True,
              help='Enables random markup of the input text at segments '
              'determined by the Unicode word breaking algorithm. For the '
              'meaning of each possible value see '
              'https://docs.gtk.org/Pango/pango_markup.html.')
@click.option('--random-markup-probability', default=0.0, show_default=True,
              help='Probabilty of random markup on segments. Set to 0 to disable random markup.',)
@click.option('--skip-unrenderable/--ignore-unrenderable',
              default=True,
              help='Skips rendering if the text contains unrenderable glyphs.')
@click.option('--line-spacing', default=None, type=click.FLOAT,
              help='Additional space between lines in points. None for default.')
@click.option('--padding-all', default=None, type=click.FLOAT,
              help='Padding in mm applied to all sides of bounding boxes and baselines.')
@click.option('--padding-horizontal', default=None, type=click.FLOAT,
              help='Padding in mm applied to left and right sides of bounding boxes and baselines.')
@click.option('--padding-vertical', default=None, type=click.FLOAT,
              help='Padding in mm applied to top and bottom sides of bounding boxes.')
@click.option('--padding-left', default=None, type=click.FLOAT,
              help='Padding in mm applied to left side of bounding boxes and baselines.')
@click.option('--padding-right', default=None, type=click.FLOAT,
              help='Padding in mm applied to right side of bounding boxes and baselines.')
@click.option('--padding-top', default=None, type=click.FLOAT,
              help='Padding in mm applied to top side of bounding boxes.')
@click.option('--padding-bottom', default=None, type=click.FLOAT,
              help='Padding in mm applied to bottom side of bounding boxes.')
@click.option('--padding-baseline', default=None, type=click.FLOAT,
              help='Padding in mm applied to left and right endpoints of baselines only.')
@click.option('--use-polygons', is_flag=True, default=False,
              help='Extract exact polygon coordinates instead of rectangular bounding boxes.')
@click.option('--use-fine-contours', is_flag=True, default=False,
              help='Prepare for fine contours (excludes --use-polygons). Emits bbox ALTO and extra _nobg.pdf.')
@click.argument('docs',
                type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
                nargs=-1)
def render(ctx,
           paper_size: tuple[int, int],
           margins: tuple[int, int, int, int],
           font: str,
           language: str,
           base_dir: Optional[Literal['L', 'R']],
           output_dir: 'PathLike',
           markup: bool,
           random_markup: list[str],
           random_markup_probability: float,
           skip_unrenderable: bool,
           line_spacing: Optional[float],
           padding_all: Optional[float],
           padding_horizontal: Optional[float],
           padding_vertical: Optional[float],
           padding_left: Optional[float],
           padding_right: Optional[float],
           padding_top: Optional[float],
           padding_bottom: Optional[float],
           padding_baseline: Optional[float],
           use_polygons: bool,
           use_fine_contours: bool,
           docs):
    """
    Renders text files into PDF documents and creates parallel ALTO facsimiles.
    """
    # Validation: can't use both polygon methods
    if use_polygons and use_fine_contours:
        raise click.BadOptionUsage('--use-polygons', 
                                 'Cannot use both --use-polygons and --use-fine-contours. Choose one.')
    
    output_dir.mkdir(exist_ok=True)

    with Pool(ctx.meta['workers'], maxtasksperchild=1000) as pool, Progress() as progress:
        render_task = progress.add_task('Rendering', total=len(docs), visible=True)
        for _ in pool.imap_unordered(partial(_render_doc,
                                             output_dir=output_dir,
                                             paper_size=paper_size,
                                             margins=margins,
                                             font=font,
                                             language=language,
                                             base_dir=base_dir,
                                             enable_markup=markup,
                                             random_markup=random_markup,
                                             random_markup_probability=random_markup_probability,
                                             skip_unrenderable=skip_unrenderable,
                                             line_spacing=line_spacing,
                                             padding_all=padding_all,
                                             padding_horizontal=padding_horizontal,
                                             padding_vertical=padding_vertical,
                                             padding_left=padding_left,
                                             padding_right=padding_right,
                                             padding_top=padding_top,
                                             padding_bottom=padding_bottom,
                                             padding_baseline=padding_baseline,
                                             use_polygons=use_polygons,
                                             use_fine_contours=use_fine_contours), docs):
            progress.update(render_task, total=len(docs), advance=1)


def _rasterize_doc(inp, output_base_path, dpi, use_polygons, use_fine_contours, 
                   smear_start, smear_inc, include_all_pixels, padding):
    from pangolinos.rasterize import rasterize_document
    rasterize_document(doc=inp[0],
                       output_base_path=output_base_path,
                       writing_surface=inp[1],
                       dpi=dpi,
                       use_polygons=use_polygons,
                       use_fine_contours=use_fine_contours,
                       smear_start=smear_start,
                       smear_inc=smear_inc,
                       include_all_pixels=include_all_pixels,
                       padding=padding)


@cli.command('rasterize')
@click.pass_context
@click.option('-d', '--dpi', default=300, show_default=True,
              help='Resolution for PDF rasterization.')
@click.option('-O', '--output-dir',
              type=click.Path(exists=False,
                              dir_okay=True,
                              file_okay=False,
                              writable=True,
                              path_type=Path),
              show_default=True,
              default=Path('.'),
              help='Base output path to place image and rewritten XML files into.')
@click.option('-w', '--writing-surface',
              type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
              default=None,
              multiple=True,
              help='Image file to overlay the rasterized text on. If multiple '
              ' are given a random one will be selected for each input file.')
@click.option('-W', '--surface-files',
              default=None,
              multiple=True,
              callback=_validate_manifests,
              type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to background images data')
@click.option('--use-polygons', is_flag=True, default=False,
              help='Process polygon coordinates instead of rectangular bounding boxes.')
@click.option('--use-fine-contours', is_flag=True, default=False,
              help='Process fine contours using image processing (excludes --use-polygons).')
@click.option('--smear-start', default=100, type=int,
              help='Start value for smearing kernel (pixels). Only used with --use-fine-contours.')
@click.option('--smear-inc', default=100, type=int,
              help='Increment for smearing kernel (pixels). Only used with --use-fine-contours.')
@click.option('--include-all-pixels', is_flag=True, default=False,
              help='Include all black pixels inside bbox. Only used with --use-fine-contours.')
@click.option('--padding', default=4, type=int,
              help='Padding in pixels around final contour. Only used with --use-fine-contours.')
@click.argument('docs',
                type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
                nargs=-1)
def rasterize(ctx,
              dpi: int,
              output_dir: 'PathLike',
              writing_surface,
              surface_files,
              use_polygons: bool,
              use_fine_contours: bool,
              smear_start: int,
              smear_inc: int,
              include_all_pixels: bool,
              padding: int,
              docs):
    """
    Accepts ALTO XML files created with `pangolinos render`, rasterizes PDF
    files linked in them with the chosen resolution, and rewrites the physical
    coordinates in the ALTO to the rasterized pixel coordinates.
    """
    # Validation: can't use both polygon methods
    if use_polygons and use_fine_contours:
        raise click.BadOptionUsage('--use-polygons', 
                                 'Cannot use both --use-polygons and --use-fine-contours. Choose one.')
    
    output_dir.mkdir(exist_ok=True)

    if not writing_surface:
        writing_surface = []

    if surface_files:
        writing_surface.extend(surface_files)

    if writing_surface:
        from random import choices
        writing_surface = choices(writing_surface, k=len(docs))

    docs = list(zip_longest(docs, writing_surface))

    # Temporarily disable multiprocessing for testing
    from pangolinos.rasterize import rasterize_document
    with Progress() as progress:
        rasterize_task = progress.add_task('Rasterizing', total=len(docs), visible=True)
        for doc in docs:
            rasterize_document(doc=doc[0],
                             output_base_path=output_dir,
                             writing_surface=doc[1],
                             dpi=dpi,
                             use_polygons=use_polygons,
                             use_fine_contours=use_fine_contours,
                             smear_start=smear_start,
                             smear_inc=smear_inc,
                             include_all_pixels=include_all_pixels,
                             padding=padding)
            progress.update(rasterize_task, total=len(docs), advance=1)


def _polygonize_doc(inp, output_dir, format_type, topline, scale):
    from pangolinos.polygonize import polygonize_document
    polygonize_document(doc=inp,
                       output_base_path=output_dir,
                       format_type=format_type,
                       topline=topline,
                       scale=scale)


@cli.command('polygonize')
@click.pass_context
@click.option('-f', '--format-type', type=click.Choice(['alto', 'page']), default='page',
              help='Sets the input document format. Default is PAGE-XML.')
@click.option('-tl', '--topline', 'topline', flag_value='topline',
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline.')
@click.option('-cl', '--centerline', 'topline', flag_value='centerline')
@click.option('-bl', '--baseline', 'topline', flag_value='baseline', default=True)
@click.option('--scale', show_default=True, type=click.INT, default=1800, help='A integer height '
              'containing optional scale factors of the input. Default 1800.')
@click.option('-O', '--output-dir',
              type=click.Path(exists=False,
                              dir_okay=True,
                              file_okay=False,
                              writable=True,
                              path_type=Path),
              show_default=True,
              default=Path('.'),
              help='Base output path to place polygonized XML files into.')
@click.argument('docs',
                type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
                nargs=-1)
def polygonize(ctx,
               format_type: str,
               topline: Optional[str],
               scale: int,
               output_dir: 'PathLike',
               docs):
    """
    Polygonizes text lines in PAGE-XML or ALTO files using the complete kraken algorithm.
    """
    output_dir.mkdir(exist_ok=True)

    with Pool(ctx.meta['workers'], maxtasksperchild=1000) as pool, Progress() as progress:
        polygonize_task = progress.add_task('Polygonizing', total=len(docs), visible=True)
        for _ in pool.imap_unordered(partial(_polygonize_doc,
                                           output_dir=output_dir,
                                           format_type=format_type,
                                           topline=topline,
                                           scale=scale), docs):
            progress.update(polygonize_task, total=len(docs), advance=1)
