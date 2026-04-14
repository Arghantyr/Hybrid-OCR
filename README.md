# About
Welcome to the testing ground for OCR tools! This place is meant to host deterministic-indeterministic hybrids to get the most of the new technology while not forgetting about the predictability, stability and speed of non-neural net solutions.

# Content
Most of the subprojects will have a working Jupyter Notebook example aside of the code base.
## PyTesseract-GLM.OCR
Originally prepared for the Kaggle's "Deep Past Challenge 2025" to handle difficult Old Assyrian transliterations and its corresponding English translations. The codebase was curated with Gemini 3 Flash and Gemini 3.1 PRO to break the high resolution images of the book pages into individual sentences to decrease the memory footprint and work with high accuracy on a Testa T4 GPU. the code base is messy but workable.
### Known issues
- sections reconstruction fails for colum sections broken between the pages
- spots/dirt is mistakenly counted as a valid character and bloats the blocks
- GLM-OCR hallucinations: occassionally the model takes the OCR'ed text as input and will try to resolve it, e.g., "Comment" will prompt it to enter a self-comment loop without any sense
- GLM-OCR artifacts: for letters with diacritics, sub/superscripts the model will inconsistently assign HTML or LaTeX formatting
