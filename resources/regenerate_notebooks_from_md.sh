for f in ../*/*.md
do
    jupytext --to notebook "$f" 
done
