# Pangolinos - PangoLine on Steroids

**Pangolinos** is an enhanced fork of the original **PangoLine** tool, providing advanced features for rendering text into PDF documents and creating parallel ALTO files with precise polygon coordinates.

This project extends the original PangoLine functionality with:
- **Exact polygon extraction** instead of rectangular bounding boxes
- **Advanced padding controls** for baselines and bounding boxes  
- **Line spacing customization** for better text layout
- **Polygonization integration** for complex document processing
- **Enhanced font resolution** with cross-platform compatibility

Pangolinos is intended to support the rendering of most of the world's writing systems
in order to create synthetic page-level training data for automatic text
recognition systems with pixel-perfect accuracy.

> **Note**: This is a community fork of the original [pangoline-tool](https://github.com/mittagessen/pangoline) by Benjamin Kiessling. All original functionality is preserved while adding significant enhancements.

## Installation

**Pangolinos** requires PyGObject and the Pango/Cairo libraries on your system. Due to PyGObject compatibility issues with conda, this project is only supported via Python virtual environments.

### System Requirements

- **Linux distributions** supporting PyGObject>=3.54.3 (tested on Debian/sid)
- **Python 3.9+** in a virtual environment
- **fc-match** system utility for font resolution

### Installation from Source

Clone the repository and install in a virtual environment:

    ~> git clone https://github.com/johnlockejrr/pangolinos.git
    ~> cd pangolinos
    ~> python -m venv venv
    ~> source venv/bin/activate
    ~> pip install -e .

### Dependencies

Install system dependencies (Debian/Ubuntu):

    ~> sudo apt-get install python3-gi python3-gi-cairo gir1.2-pango-1.0 libpango1.0-dev libcairo2-dev fontconfig

All Python dependencies are automatically installed via pip.

## Usage

### Rendering

Pangolinos renders text first into vector PDFs and ALTO facsimiles using some
configurable "physical" dimensions.

    ~> pangolinos render doc.txt
    Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Various options to direct rendering such as page size, margins, language, and
base direction can be manually set, for example:

    ~> pangolinos render -p 216 279 -l en-us -f "Noto Sans 24" doc.txt

Text can also be styled with [Pango
Markup](https://docs.gtk.org/Pango/pango_markup.html). Parsing is disabled per
default but can be enabled with a switch. You'll need to escape any characters
that are part of XML such as &, <, >, quotes, and various control characters
using [HTML
entities](https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references).

    ~> pangolinos render --markup doc.txt

It is possible to randomly insert stylization of Unicode [word
segments](https://unicode.org/reports/tr29/#Word_Boundaries) in the text. One
or more styles will be randomly selected from a configurable list of styles:

    ~> pangolinos render --random-markup-probability 0.01 doc.txt

The probability is the probability of at least one style being applied to any
particular segment. A subset of the total available number of styles is enabled
by default when a probability greater than 0 is given. To change the list of
possible styles:

    ~> pangolinos render --random-markup-probability 0.01 --random-markup style_italic --random-markup variant_smallcaps doc.txt

The semantics of each value can be found in the [pango documentation](https://docs.gtk.org/Pango/pango_markup.html).

Styling with color is treated slightly differently than other styles. In
general, colors are selected with the `foreground_*` style. As a large number
of colors are known to Pango, the `foreground_random` alias exists that enables
all possible colors:

    ~> pangolinos render  --random-markup-probability 0.01 --random-markup foreground_random doc.txt

When applying random styles to words, control characters in the source text
should *not* be escaped as pangoline internally escapes any characters that
require it.

### Rasterization

In a second step those vector files can be rasterized into PNGs and the
coordinates in the ALTO files scaled to the selected resolution (per default
300dpi):

    ~> pangolinos rasterize doc.0.xml doc.1.xml ...
    Rasterizing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Rasterized files and their ALTOs can be used as is as ATR training data.

To obtain slightly more realistic input images it is possible to overlay the
rasterized text into images of writing surfaces.

    ~> pangolinos rasterize -w ~/background_1.jpg doc.0.xml doc.1.xml ...

Rasterization can be invoked with multiple background images in which case they
will be sampled randomly for each output page. A tarball with 70 empty paper
backgrounds of different origins, digitization qualities, and states of
preservation can be found [here](http://l.unchti.me/paper.tar).

For larger collections of texts it is advisable to parallelize processing,
especially for rasterization with overlays:

    ~> pangolinos --workers 8 render *.txt
    ~> pangolinos --workers 8 rasterize *.xml

### Enhanced Features

#### Polygon Extraction
Extract precise polygon coordinates instead of rectangular bounding boxes:

    ~> pangolinos render --use-polygons -l he-il -f "Shlomo Stam" document.txt --output-dir out-poly
    ~> pangolinos rasterize --use-polygons out-poly/*.xml

#### Fine Contours (Aletheia‑style)
Compute tight “fine contour” polygons for each TextLine starting from bounding boxes. This follows an Aletheia‑like smearing workflow over a binarized, background‑free raster while keeping your final page image (with or without background) intact.

Workflow:

- Render (bbox ALTO only):

        ~> pangolinos render --use-fine-contours -l he-il -f "Noto Serif Hebrew 18" doc.txt -O out_fine

  - Produces: `out_fine/doc.0.pdf` and `out_fine/doc.0.xml` (ALTO with bounding boxes).
  - No special contour options here; this step prepares standard ALTO/PDF.

- Rasterize and replace each line’s bbox with a fine contour polygon:

        # With backgrounds
        ~> pangolinos rasterize --use-fine-contours \
             --smear-start 100 --smear-inc 100 --padding 4 \
             -W backgrounds.lst -O out_fine_rast out_fine/doc.0.xml

        # Without backgrounds
        ~> pangolinos rasterize --use-fine-contours \
             --smear-start 100 --smear-inc 100 --padding 4 \
             -O out_fine_rast out_fine/doc.0.xml

What happens under the hood:

- If backgrounds are used (`-w` or `-W`), rasterize writes the final page PNG with the background and also renders a clean page PNG (no background) solely for contour extraction. If no background is used, the main page image is reused for contouring.
- The clean image is binarized (BW), smearing is applied horizontally to merge glyphs into a single line component, and the outer contour is extracted and written back into ALTO as `<Shape><Polygon POINTS="..."/>` per `TextLine`.
- Temporary clean/BW images are removed after processing.

Options (rasterize only):

- `--smear-start INTEGER`: Initial horizontal kernel (pixels) for smearing (default: 100)
- `--smear-inc INTEGER`: Increment for the kernel (pixels) across iterations (default: 100)
- `--include-all-pixels`: Use all dark pixels in the bbox (bypass component selection)
- `--padding INTEGER`: Extra dilation applied to the final outline (pixels; default: 4)

Notes:

- Fine contours operate in pixel space during rasterization; ALTO is temporarily scaled to pixels for accurate region cropping, then polygons are written back to the output ALTO.
- This feature requires OpenCV; `opencv-python-headless` is declared in `setup.cfg`.

#### Advanced Padding Controls
Fine-tune baselines and bounding boxes with 8 padding options:

    ~> pangolinos render --padding-all 2.0 --padding-baseline 1.5 document.txt
    ~> pangolinos render --padding-left 3.0 --padding-right 1.5 --padding-top 2.0 document.txt

#### Line Spacing Customization
Control line spacing for better text layout:

    ~> pangolinos render --line-spacing 2.4 document.txt

#### Polygonization Integration
Process complex documents with integrated polygonization. Pangolinos can repolygonize ALTO-XML files generated with pangoline/pangolinos that contain bounding boxes, converting them to precise polygon coordinates:

    ~> pangolinos polygonize document.xml --format-type alto --scale 2000
    ~> pangolinos polygonize --format-type alto *.xml

## Limitations

In order to achieve proper typesetting quality, Pango requires placing the
whole text on a single layout before splitting it into individual pages by
translating each line of the layout onto a page surface. This approach limits
the maximum print space of a single text to 739.8 meters, roughly 3000 pages
depending on paper size and margins, before an overflow of the 32 bit integer
baseline position y-offset will occur.

## Credits

**Pangolinos** is a community fork of the original [pangoline-tool](https://github.com/mittagessen/pangoline) by [Benjamin Kiessling](https://github.com/mittagessen).

### Original Project
- **Author**: Benjamin Kiessling
- **Repository**: https://github.com/mittagessen/pangoline
- **License**: Apache-2.0

### Enhanced Features
- **Polygon extraction system** with PangoCairo integration
- **Advanced padding controls** for precise coordinate adjustment
- **Line spacing customization** for improved text layout
- **Polygonization integration** for complex document processing
- **Enhanced font resolution** with cross-platform compatibility

## Funding

<table border="0">
 <tr>
    <td> <img src="https://raw.githubusercontent.com/mittagessen/kraken/main/docs/_static/normal-reproduction-low-resolution.jpg" alt="Co-financed by the European Union" width="100"/></td>
    <td>The original PangoLine project was funded in part by the European Union. (ERC, MiDRASH, project number 101071829).</td>
 </tr>
</table>
