To install dependencies using `conda`:

    conda create -n eedp python=3.6
    source activate eedp
    conda install cython numpy nltk lxml
    pip install git+https://github.com/clab/dynet#egg=dynet
    pip install git+https://github.com/myedibleenso/py-processors.git@clu-bio
