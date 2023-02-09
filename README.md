# LatentSimilarity
Metric learning-based predictive model for small, high dimensional datasets

<img src='https://raw.githubusercontent.com/aorliche/LatentSimilarity/main/images/overview.png' alt='LatSim overview' width='600'>

A. Orlichenko et al., "Latent Similarity Identifies Important Functional Connections for Phenotype Prediction," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2022.3232964.

## Capabilities
- Very fast runtime
- High accuracy on limited data
- Multimodal
- Sklearn interface

## Requirements
- python
- pytorch with cuda
- numpy
- sklearn
- requests (to get sample data)

## Usage
Take a look at the example in the ```notebooks``` directory for sample usage.

    from sklearn.model_selection import train_test_split
    from latsim import LatSimClf

    ...

    xtr, xt, ytr, yt = train_test_split(x, y, stratify=y, train_size=0.75)

    clf = LatSimClf().fit(xtr,ytr,ld=1)
    yhat = clf.predict(xt)

An interactive demo was available <a href='https://aorliche.github.io/LatSim/'>here</a>.
We are working to put up another version.

## Contact
Anton Orlichenko | aorlichenko@tulane.edu<br/>
<a href='https://aorliche.github.io/'>aorliche.github.io</a>
<a href='https://www2.tulane.edu/~wyp/'>MBB Laboratory</a>
