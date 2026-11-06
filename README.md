# GameChallengeTools-py
Python version of https://github.com/Iannyck/GameChallengeTools

This is a demo of the algorithm presented in the paper A Comprehensive Model of Automated Evaluation of Difficulty in Platformer Games. This project includes helper function to extrapolate information images and data from mario levels to then process into a difficulty curve. You can find an example implementation in python of the actual algorithm in gamedifficulty/Processing.py, function CalculateDifficulty

![example difficulty curve](example_diff_1_1_320.png)

## Batch level analysis

Run `python main.py` to analyze level images in parallel. Place images (PNG, JPEG,
BMP, TIFF, or WebP) under `ressources/levels/`; subdirectories are searched
recursively. The generated difficulty graphs and analysis images are saved next
to each source image, with the source filename and extension as a prefix.

For compatibility with the original repository layout, if `ressources/levels/`
does not exist, the script recursively processes files named `level.png` under
`ressources/`.

The analysis requires exactly one source image inside a folder named
`Niveau_1_1`. It uses that image as the reference and writes
`difficulty_area_comparison.csv` at the root of `ressources/levels/`. The CSV
contains the image path, its parent folder in the `algorithm` column, the
normalized trapezoidal area under the reference and comparison curves, their
signed and absolute area difference, and the integral of the absolute
difference between the two profiles. It uses semicolons as column delimiters
and commas as decimal separators for compatibility with French Excel settings.
Curves are compared over normalized level progress, so their widths do not
affect the area values. Pacing columns count local difficulty peaks with a
minimum prominence of `0.05`; peaks less than one difficulty-window apart are
merged to avoid counting the same spike multiple times. Pacing is reported as
the mean distance between peaks in pixels and the number of peaks per 1,000
level pixels, both for the image and the `Niveau_1_1` reference. Mean spacing
is left empty when there are fewer than two peaks.