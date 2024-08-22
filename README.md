# Evaluating Factuality Word Definitions

## Setup

Clone it via:
```
git clone --recurse-submodules git@github.com:lukasellinger/evaluating_factuality_word_definitions.git
```
To set up the project environment, follow these steps:

1. Install the required Python packages and models by running:
```
pip install -r requirements.txt
python setup.py
```
2. Setup config file and add necessary attributes:
```
cp config.py.template config.py
nano config.py
```

3. Repository for Discourse Simplification
```
cd ..
git clone git@github.com:Lambda-3/DiscourseSimplification.git
cd DiscourseSimplification
git checkout 5e7ac12
mvn clean install -DskipTests
```

## Authors and acknowledgment
Lukas Ellinger (lukas.ellinger@tum.de)