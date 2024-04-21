# When performing the docker build move into docker folder /docker
#  docker build -t project_image .
FROM python:3.11-slim

RUN pip install poetry

#Here you are changing/creating your work directory, so now you have to work there.
WORKDIR /opt/app

#You copy all the files of your local directory to your work directory
COPY . /opt/app

RUN poetry install

#Before running the uvicorn you have to change the directory to project_code, as you did everytime when using the terminal
WORKDIR /opt/app/project_code

EXPOSE 8000

CMD [ "poetry", "run", "uvicorn", "project_code.app:app", "--host", "0.0.0.0"]