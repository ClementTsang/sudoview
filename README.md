## sudoview

A program that uses OpenCV and PyTesseract to solve Sudoku boards.

## Installation

Just clone the repository, and run sudoview.py.  Requires the PyTesseract and OpenCV libraries.  If you use pip:
```
pip install pillow
pip install pytesseract
pip install opencv-python
```

## Usage

Call the python script with the following:

```
python sudoview.py -i "IMAGE_PATH" -s/--solve [true/false]
```

The -s/--solve flag is optional, if you wanted to use this to somehow redirect the scanned grid to, say, a text file, and not run the backtrack solver.
