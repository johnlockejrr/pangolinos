# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-10-08

### Added

#### Polygonize Integration
- **Integrated `polygonize.py` script into pangoline CLI** - Added `pangoline polygonize` command
- **Moved `polygonize.py`** from root directory to `pangoline/` subdirectory for proper module structure
- **Refactored polygonize functionality** to follow pangoline's module patterns:
  - Removed standalone CLI functionality
  - Created clean `polygonize_document()` function for CLI integration
  - Added proper type hints and error handling
  - Maintained all original kraken algorithm functionality
- **Added polygonize CLI command** with full option support:
  - `-f, --format-type [alto|page]` - Input document format (default: page)
  - `-tl, --topline` - For hanging baseline data
  - `-cl, --centerline` - For centerline data  
  - `-bl, --baseline` - For baseline data (default)
  - `--scale INTEGER` - Scale factor (default: 1800)
  - `-O, --output-dir DIRECTORY` - Output directory (default: current directory)
- **Added parallel processing support** for polygonize command using the same pattern as render and rasterize

#### Line Spacing Enhancement
- **Added `--line-spacing` option** to render command for controlling line spacing
- **Added `line_spacing` parameter** to `render_text()` function
- **Added line spacing logic** using `layout.set_spacing()` with proper Pango scaling
- **Added comprehensive documentation** for line spacing functionality

#### Comprehensive Padding System
- **Added 8 padding options** for precise control over baselines and bounding boxes:
  - `--padding-all` - Padding applied to all sides of bounding boxes and baselines
  - `--padding-horizontal` - Padding applied to left and right sides of bounding boxes and baselines
  - `--padding-vertical` - Padding applied to top and bottom sides of bounding boxes
  - `--padding-left` - Padding applied to left side of bounding boxes and baselines
  - `--padding-right` - Padding applied to right side of bounding boxes and baselines
  - `--padding-top` - Padding applied to top side of bounding boxes
  - `--padding-bottom` - Padding applied to bottom side of bounding boxes
  - `--padding-baseline` - Padding applied to left and right endpoints of baselines only
- **Added additive padding system** - Multiple padding options can be combined
- **Added proper unit conversion** - Padding values in mm are correctly converted to points
- **Added comprehensive documentation** for all padding parameters
- **Added padding logic** to coordinate calculations in `render_text()` function

#### Polygon Extraction System
-
#### Fine Contours (Aletheia‑style)
- **Added `--use-fine-contours` mode**: two‑stage pipeline that keeps render simple (bbox ALTO) and computes fine contours during rasterize.
- **Render (prepare only)**: `pangolinos render --use-fine-contours` emits standard bbox ALTO and normal PDFs (no special parameters here).
- **Rasterize (contour computation)**: `pangolinos rasterize --use-fine-contours` computes a tight polygon per `TextLine` using a smearing strategy over a clean, binarized page image and writes `<Shape><Polygon>` back into ALTO.
- **Rasterize‑only options**:
  - `--smear-start INTEGER` – initial horizontal kernel size (px)
  - `--smear-inc INTEGER` – increment per iteration (px)
  - `--include-all-pixels` – bypass component selection and include all dark pixels
  - `--padding INTEGER` – extra dilation of the final outline (px)
- **Background‑aware workflow**: when rasterizing with `-w/-W` writing surfaces, an additional clean (no‑background) PNG is rendered internally only for contour extraction; otherwise the main page image is reused. Temporary files are cleaned up.
- **OpenCV‑based implementation**: binarization + horizontal smearing + morphological closing + external contour extraction for a single, line‑wide envelope.

- **Added `--use-polygons` option** to both render and rasterize commands for exact polygon coordinates
- **Implemented PangoCairo-based polygon extraction** using `layout_line_path()` for accurate glyph outlines
- **Added FreeType font resolution** with programmatic font path detection using `fc-match`
- **Created simplified polygon processing** that preserves character shapes while connecting glyphs
- **Added ALTO XML polygon template** (`alto-polygons.tmpl`) for proper polygon output format
- **Implemented coordinate system transformation** from FreeType font units to page coordinates
- **Added robust error handling** for complex glyph shapes and font loading failures
- **Created visualization tools** for debugging polygon extraction results

### Technical Details

#### Polygonize Integration
- **File changes:**
  - Moved `polygonize.py` → `pangoline/polygonize.py`
  - Updated `pangoline/cli.py` to include polygonize command
  - Removed original `polygonize.py` from root directory
- **Function signatures:**
  - Added `polygonize_document()` function with proper type hints
  - Added `_polygonize_doc()` helper function for multiprocessing
  - Added `_replace_polygons_in_xml()` and related helper functions
- **CLI integration:**
  - Added `@cli.command('polygonize')` decorator
  - Added all polygonize options with proper click types
  - Integrated with existing multiprocessing system

#### Line Spacing Enhancement
- **File changes:**
  - Updated `pangoline/render.py` to include line spacing parameter
  - Updated `pangoline/cli.py` to include line spacing option
- **Implementation:**
  - Added `line_spacing: Optional[float] = None` parameter
  - Added `layout.set_spacing(int(line_spacing * Pango.SCALE))` logic
  - Added proper documentation and type hints

#### Padding System
- **File changes:**
  - Updated `pangoline/render.py` to include 8 padding parameters
  - Updated `pangoline/cli.py` to include 8 padding options
- **Implementation:**
  - Added padding calculation logic in coordinate processing section
  - Added proper mm to points conversion using `_mm_point` constant
  - Added additive padding system allowing multiple options to be combined
  - Added comprehensive parameter documentation
- **Coordinate modification:**
  - Applied padding to `left`, `right`, `top`, `bottom` coordinates
  - Applied baseline padding to left and right endpoints
  - Maintained proper coordinate bounds and rounding

#### Polygon Extraction System
#### Fine Contours
- **File changes:**
  - Created `pangoline/fine_contour_extractor.py` (OpenCV pipeline for fine contours)
  - Updated `pangoline/cli.py`:
    - Render: added `--use-fine-contours` (prepare mode only; no smear params here)
    - Rasterize: added `--use-fine-contours` and options `--smear-start`, `--smear-inc`, `--include-all-pixels`, `--padding`
  - Updated `pangoline/render.py`: uses bbox ALTO when `--use-fine-contours` is set; no extra `_nobg.pdf` is produced
  - Updated `pangoline/rasterize.py`:
    - Renders the main page PNG; if a background is used, also renders a clean no‑background PNG for contouring
    - Builds a temporary pixel‑scaled ALTO for accurate ROI cropping
    - Inserts `<Shape>/<Polygon>` if missing; matches polygons by `TextLine` ID
    - Cleans temporary images/ALTO after processing
- **Implementation highlights:**
  - Strict bbox‑local processing; smearing prefers horizontal connectivity to merge words without vertical bleed
  - Optional padding dilates final outline; supports including all pixels
  - Robust to MultiPolygon: always selects the largest external contour per line
- **Dependencies:** `opencv-python-headless` added in `setup.cfg`.

- **File changes:**
  - Created `pangoline/polygon_extractor.py` with PangoCairo-based extraction
  - Updated `pangoline/render.py` to support polygon extraction mode
  - Updated `pangoline/rasterize.py` to preserve polygon coordinates
  - Created `pangoline/templates/alto-polygons.tmpl` for polygon output
  - Added `--use-polygons` option to both render and rasterize commands
- **Implementation:**
  - Used `PangoCairo.layout_line_path()` to extract exact glyph outlines
  - Implemented font resolution via `fc-match` for cross-platform compatibility
  - Added coordinate transformation from FreeType font units to page coordinates
  - Created simplified polygon processing that preserves character shapes
  - Added robust error handling for complex glyph shapes and font loading
- **Key features:**
  - Extracts precise polygon coordinates instead of rectangular bounding boxes
  - Supports complex scripts (Hebrew, Arabic, etc.) with proper RTL handling
  - Maintains glyph-level accuracy while creating connected line envelopes
  - Provides visualization tools for debugging and validation
- **Dependencies:**
  - Requires `fc-match` system utility for font resolution
  - All Python dependencies are included in `setup.cfg`

### Usage Examples

#### Polygonize Command
```bash
# Basic polygonization
pangoline polygonize document.xml

# With specific format and scale
pangoline polygonize -f alto --scale 2000 document.xml

# With topline annotation
pangoline polygonize -tl document.xml

# With custom output directory
pangoline polygonize -O /path/to/output document.xml
```

#### Line Spacing
```bash
# Add 2 points of line spacing
pangoline render --line-spacing 2.0 document.txt

# Combined with other options
pangoline render --line-spacing 1.5 --font "Arial 12" document.txt
```

#### Padding Options
```bash
# Add 2mm padding to all sides
pangoline render --padding-all 2.0 document.txt

# Add horizontal padding only
pangoline render --padding-horizontal 1.5 document.txt

# Combine different padding options
pangoline render --padding-left 3.0 --padding-right 1.5 --padding-top 2.0 document.txt

# Baseline-only padding
pangoline render --padding-baseline 2.5 document.txt

# Complex padding combination
pangoline render --padding-all 1.0 --padding-left 2.0 --padding-baseline 1.5 document.txt
```

#### Polygon Extraction
pangolinos Fine Contours
```bash
# 1) Render (prepare bbox ALTO + PDF)
pangolinos render --use-fine-contours -l he-il -f "Noto Serif Hebrew 18" doc.txt -O out_fine

# 2) Rasterize and compute contours (with backgrounds)
pangolinos rasterize --use-fine-contours \
  --smear-start 100 --smear-inc 100 --padding 4 \
  -W backgrounds.lst -O out_fine_rast out_fine/doc.0.xml

# 2) Rasterize and compute contours (no backgrounds)
pangolinos rasterize --use-fine-contours \
  --smear-start 100 --smear-inc 100 --padding 4 \
  -O out_fine_rast out_fine/doc.0.xml
```
```bash
# Render with polygon extraction
pangoline render --use-polygons -l he-il -f "Shlomo Stam" document.txt --output-dir out-poly

# Rasterize with polygon preservation
pangoline rasterize --use-polygons *.xml

# Combined workflow
pangoline render --use-polygons document.txt --output-dir out-poly
pangoline rasterize --use-polygons out-poly/*.xml
```

### Backward Compatibility

- **All new parameters default to `None`** - existing code continues to work unchanged
- **No breaking changes** to existing function signatures or CLI commands
- **Maintained existing behavior** when new options are not specified
- **Preserved all original functionality** while adding new features

### Testing

- **Comprehensive testing** performed for all new functionality
- **Verified backward compatibility** with existing commands
- **Tested parameter combinations** to ensure proper additive behavior
- **Validated coordinate calculations** for accuracy and precision

---

**Contributor:** [johnlockejrr](https://github.com/johnlockejrr)  
**Date:** 2025-10-08  
**Version:** Unreleased
