find . -type f -name "*.ipynb" | while read fname
do
    jupytext --to md "${fname}"
done