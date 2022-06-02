# Overview

This project is a webservice API running thanks to FastAPI. Webservice methods allow user/app to query a model to predict tags for new questions on Stack Overflow.

Once project is running webservice method are visible on a documentation page while calling /docs endpoint.

# Requirements

To execute this app on your system you need a python version that is at least 3.8, or a running docker service.

# Install

First clone the project :
```
$ git clone https://github.com/mikachou/stackoverflow_api.git
```

## Using system-install python

To execute this app on your system you need a python version that is at least 3.8.

First install python dependencies from project folder :
```
$ pip install -r requirements.txt
```

Then to run project in dev mode go inside app sub-folder then type :
```
$ uvicorn app:app --reload
```

Once app is launched use it in browser :
http://localhost:8000

## Using docker

Build docker image :
```
$ docker build . -t stackoverflow_fastapi
```

Run a container using image :
```
$ docker run -p 8000:80 --name stackoverflow_fastapi -it stackoverflow_fastapi
```

Once app is launched use it in browser :
http://localhost:8000