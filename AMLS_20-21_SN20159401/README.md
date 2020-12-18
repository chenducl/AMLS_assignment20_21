# AMLS_20-21_SN20159401



## Organization of Project



**Important!**

When you want to run the code, please unzip the training-set in the Datasets folder and name it `train`, and unzip the test-set in the Datasets folder and name it `test`.As shown in the directory tree below.

**CUDA is necessary !!!!**



```
AMLS_ASSIGNMENT20_21
└─AMLS_20-21_SN20159401
    ├─A1
    │  └─__pycache__
    ├─A2
    │  └─__pycache__
    ├─B1
    │  └─__pycache__
    ├─B2
    │  └─__pycache__
    └─Datasets
        ├─cartoon_set
        │  └─img
        └─celeba
            └─img
```



## Role of each file



Files named `__init_.py` in A1,A2,B1,B2 make the directory as a submodule for easily import

Files named `__main__.py` in A1,A2,B1,B2 declare the model for each subtasks

`main.py`  run model for each task and print the result in console

`requirements.txt` declares the libraries install in my environment for conveniently be  installed again.

`shape_predictor_68_face_landmarks.dat` is the configuration file for HOG algorithm, which is loaded by dlib in `main.py`







## Package Requirement



In general, these libraries need to be installed

1. pytorch
2. tensorflow
3. dlib
4. numpy
5. pandas
6. matplotlib
7. etc.



*detailed pip freeze result as followed*

```
absl-py==0.11.0
appdirs==1.4.4
asn1crypto==1.2.0
astroid==2.4.2
astunparse==1.6.3
atomicwrites==1.4.0
attrs==19.3.0
beautifulsoup4==4.9.1
black==19.10b0
bs4==0.0.1
cached-property==1.5.2
cachetools==4.1.1
certifi==2020.6.20
cffi==1.13.2
chardet==3.0.4
click==7.1.2
colorama==0.4.3
conda==4.8.4
conda-package-handling==1.6.0
cryptography==2.8
cycler==0.10.0
distlib==0.3.1
dlib==19.21.0
filelock==3.0.12
flake8==3.8.3
gast==0.3.3
get==2019.4.13
google-auth==1.23.0
google-auth-oauthlib==0.4.2
google-pasta==0.2.0
grpcio==1.33.2
h5py==2.10.0
html5validator==0.3.3
idna==2.10
importlib-metadata==1.7.0
iniconfig==1.0.1
isort==4.3.21
Jinja2==2.11.2
joblib==0.17.0
Keras==2.4.3
Keras-Preprocessing==1.1.2
kiwisolver==1.2.0
lazy-object-proxy==1.4.3
Markdown==3.3.3
MarkupSafe==1.1.1
matplotlib==3.3.2
mccabe==0.6.1
menuinst==1.4.16
more-itertools==8.4.0
numpy==1.18.5
oauthlib==3.1.0
opencv-python==4.4.0.46
opt-einsum==3.3.0
packaging==20.4
pandas==1.1.2
pathlib==1.0.1
pathspec==0.8.0
Pillow==7.2.0
pluggy==0.13.1
post==2019.4.13
protobuf==3.14.0
public==2019.4.13
py==1.9.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycodestyle==2.6.0
pycosat==0.6.3
pycparser==2.19
pydocstyle==5.0.2
pyflakes==2.2.0
pylint==2.5.3
pyOpenSSL==19.1.0
pyparsing==2.4.7
PySocks==1.7.1
pytest==6.0.1
python-dateutil==2.8.1
pytz==2020.1
pywin32==223
PyYAML==5.3.1
query-string==2019.4.13
regex==2020.6.8
request==2019.4.13
requests==2.24.0
requests-oauthlib==1.3.0
rsa==4.6
ruamel-yaml==0.15.46
scikit-learn==0.23.2
scipy==1.5.2
six==1.15.0
sklearn==0.0
snowballstemmer==2.0.0
soupsieve==2.0.1
tensorboard==2.4.0
tensorboard-plugin-wit==1.7.0
tensorflow==2.3.1
tensorflow-estimator==2.3.0
termcolor==1.1.0
threadpoolctl==2.1.0
toml==0.10.1
tox==3.19.0
tqdm==4.40.0
typed-ast==1.4.1
urllib3==1.25.10
virtualenv==20.0.30
Werkzeug==1.0.1
win-inet-pton==1.1.0
wincertstore==0.2
wrapt==1.12.1
zipp==3.1.0

```

