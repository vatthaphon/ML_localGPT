import datetime
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


import shutil

def move_file(source_file, destination_dir):
    try:
        # Move the file from source to destination
        shutil.move(source_file, destination_dir)
        print(f"File '{source_file}' moved to '{destination_dir}' successfully.")
    except FileNotFoundError:
        print("Source file not found.")
    except PermissionError:
        print("Permission error: Unable to move the file.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_file(file_path):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        else:
            print(f"File '{file_path}' not found.")
    except PermissionError:
        print("Permission error: Unable to delete the file.")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_documents(source_dir: str) -> tuple[list[Document], dict]:    

    ## Load documents that are already processed
    if os.path.exists("extracted_files.dat"):
        logging.info(f"extracted_files.dat exists")

        with open("extracted_files.dat", "rb") as fp:
            ex_files = pickle.load(fp)
            fp.close()

    else:
        ex_files = {}

    ## Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:

        # Get the date modified
        timestamp = os.path.getmtime("./SOURCE_DOCUMENTS/" + file_path)

        # Convert the timestamp to a readable date
        date_modified = datetime.datetime.fromtimestamp(timestamp)

        ex_files_key = file_path + str(date_modified.date())

        # source_file = "D:/vattha/Data/Work/UtilSrcCode/ML_localGPT/SOURCE_DOCUMENTS/" + file_path

        if ex_files_key not in ex_files:
            logging.info(f"Load {file_path}")

            # destination_dir="D:/vattha/Data/Work/UtilSrcCode/ML_localGPT/tmp"

            # print(source_file)

            # move_file(source_file, destination_dir)
            
            ex_files[ex_files_key] = 1

            file_extension = os.path.splitext(file_path)[1]
            source_file_path = os.path.join(source_dir, file_path)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)
        else:
            logging.info(f"{file_path} is already exist.")
    #     else:
    #         delete_file(source_file)

    # return

    ## Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))

    # n_workers = 1

    logging.info(f"The number of ingest workers: {n_workers}")

    logging.info(f"len(paths): {len(paths)}")

    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs, ex_files


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):

    print("ingest.py: device_type is " + device_type)

    ## Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents, ex_files = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    logging.info(f"Create embeddings")

    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": device_type})
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    logging.info(f"Create db")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

    logging.info(f"Persist db")
    db.persist()
    db = None

    with open("extracted_files.dat", "wb") as fp:
        pickle.dump(ex_files, fp)
        fp.close()



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
