# QA System

This is a simple fastapi application which contains query endpoint when called responds with a relevant answer.

## Folder Structure

```
RAGBot
├── Chatbot                          -> Main Folder containing the code
│    ├── API                                -> Code related to Rest API in fastapi
│    │    ├── app.py                        -> FastAPI Initialisation and lifespan manager initialisation
│    │    └── settings.py                   -> settings.py containing all the environment variables loaded in to settings object
│    ├── Data                               -> Folder containing Code related to data operation
│    │    ├── document_loaders.py           -> Contains Class which loads files from given path, preprocess it
│    │    ├── embedding_based_retriver.py   -> contains class which is used to embed document using openaiembeddings and index in elasticsearch
│    │    ├── keyword_match_retriever.py    -> contains class which is used to index in elasticsearch
│    │    ├── knowledge_graph_retriever.py  -> Graph based context retriever using spacy, openAIembeddings and networkx
│    │    └── query_helper.py               -> Containing class which contains method to invoke the final chain
│    ├── Utils                              -> Folder containing files for utils used across repository
│    │    ├── utils.py                      -> logger class is defined here
│    │    └── validation.py                 -> Contains pydantic models used for validation
│    ├── .env.example                       -> example env
│    └── asgi.py                            -> file to run fastapi localy (python3 -m asgi)
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yaml                     -> docker-compose file containing elasticsearch server and backend app
├── Dockerfile                              -> Dockerfile to build the application
├── poetry.lock                             -> final resolved dependencies
├── pyproject.toml                          -> dependency manager
└── README.md                               -> Readme file explaining the folder structure and setting up the application.
```

## How to SetUp

1. Ensure you have docker on your system and running
2. Clone the repository and cd into Chatbot outer folder which will be the working directory
3. copy .env.example to .env and make necessary changes
4. `docker build -t chatbot . ` will copy and build the docker image for backend
5. `docker-compose --env-file ./Chatbot/.env up` will build and run both elasticsearch server service and backend service.
6. Run `python Chatbot/evaluater.py` to run evaluation once the services are up
