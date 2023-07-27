import json
from typing import Any, Dict, Union
import numpy as np
import random
import json

from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

from loguru import logger


import json
import numpy as np
import json

from anytree import Node
from pathlib import Path

from loguru import logger

class TaxonomyParser:

    """
    This class is simply used to parse the mapping into a python dictionary
    and provide bi-directional mapping from category index to description
    """

    def __init__(self, input_file=None):
        self.taxo_mappings = {}
        self.reverse_taxo_mappings = {}
        self.input_file = None
        self.n_levels = -1
        try:
            p = Path(input_file).resolve(strict=True)

            logger.info(f'Reading taxonomy from {p}')

            with open(p, "r") as f:
                self.taxo_mappings = json.load(f)
        
            # make the keys integer
            self.taxo_mappings = {int(k): v for k,v in self.taxo_mappings.items()}

            self.n_levels = len(self.taxo_mappings)
            self.reverse_taxo_mappings = {v: k for k, v in self.taxo_mappings.items()}

        except FileNotFoundError as fnf:
          logger.error(fnf)
          raise fnf
        
    
class Document:
    """
    A utility class that represents a document.
    We use the same Document representation for "training" and "test" documents

    The difference between a training and test document is mainly in the available fields:
        1. A training Document has 'labels', whereas a test Document does not
        2. The content of a training or test document is assumed to be a dictionary as specified in the
           json schema of train_data.json and predict_data.json.
           However if the content is a simple string, we assume that the user
           supplied us with raw text on which to do prediction
    """

    @staticmethod
    def get_datetime(date_str):
        """
        A utility function to convert a datetime string into a datetime object
        """
        date_str_cleaned = date_str.split('+')
        if len(date_str_cleaned) == 1:
            try:
                timestamp = datetime.strptime(date_str_cleaned[0], '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                timestamp = datetime.strptime(date_str_cleaned[0], '%Y-%m-%dT%H:%M:%S')
        elif len(date_str_cleaned) == 2:
            try:
                assert date_str_cleaned[1] == "00:00"
                timestamp = date_str_cleaned[0]
            except AssertionError:
                print(date_str)

        return timestamp

    def __init__(self, json_element: Union[Dict[str, Any], str], taxo: TaxonomyParser = None):
        """
        We either provide content following the schema in the challenge
        or a raw input in which case we simply take it as it is       
        """       
        self.title = ""
        self.content = ""
        # does the Document contain raw text or a payload as given in the json schema of the challenge
        self.raw_text = True  # by default assume it's raw text
        if isinstance(json_element, str):
           # this is a raw text 
            self.content = content.replace("\n", "").replace("\t", "")
        
        elif isinstance(json_element, dict):
            if metadata := json_element.get("metadata", []):
                # this is a text following the schema of the challenge
                # convert to datetime object, just to keep it around, maybe can be made use of at some later point
                self.publishedAt = Document.get_datetime(metadata['publishedAt'])

            try:
                content = json_element["content"]
            except KeyError as ke:
                # intentionally catch it to
                logger.error(ke)
                raise ke
            # a train or test document following the json schema of the challenge
            self.raw_text = False  # we know this is not raw text
            if title := content.get("title", ""):
                self.title = title.replace("\n", "").replace("\t", "")
            if fullTextHTML := content.get("fullTextHtml", ""):
                # we strip the html tags and remove a few annoying characters
                self.content = ''.join(
                    BeautifulSoup(fullTextHTML, "html.parser").text.replace("\n", "").replace("\t", ""))

            # let's store the Sections, too, may be useful
            self.sections = content.get("sections", [])

            self.labels = []
            self.raw_labels = json_element.get("labels", [])
            # build a dense vector with label predictions
            # [["crime, law and justice", 0.7524], ["unrest, conflicts and war", 0.7513], ["politics", 0.6477]]
            if self.raw_labels:
                self.labels = [0] * taxo.n_levels
                for l, confidence in self.raw_labels:
                    self.labels[taxo.reverse_taxo_mappings[l]] = confidence

    def get_text(self):
        """
        Get the raw text of a Document. This is simply a concatenation of the Title and the Content.
        """
        if self.raw_text:
            return self.content
    
        if self.title != "":
            # add the title as the first sentence in the text
            return self.title + "." + self.content
        return self.content

        

class TextCorpus:
    """
    The TextCorpus represents a collection of Documents (training and test),
    as well as, a Taxonomy mapping used to process the documents
    """
    def __init__(self, train_data: str = None, predict_data: str = None, taxo_map: str = None):
        
        self.taxo = TaxonomyParser(input_file=taxo_map)
        self.train_f = None
        self.predict_f = None
        self.train_documents = []
        self.predict_documents = []
        # train data is optional
        if train_data is not None:     
            try:
                self.train_f = Path(train_data).resolve(strict=True)                      
            except FileNotFoundError as fnf:
                logger.error(fnf)
                raise fnf
            logger.info(f"Reading training data from {self.train_f}")
            # don't catch Exceptions here, we want to stop
            json_collection = json.load(open(self.train_f, "r")) 
            # remove training documents without labels
            self.train_documents = [Document(doc, self.taxo) for doc in json_collection if doc.get("labels", [])]
    
        # test data is required
        try:
            self.predict_f = Path(predict_data).resolve(strict=True)          
        except FileNotFoundError as fnf:
            logger.error(fnf)
            raise fnf

        #
        logger.info(f"Reading test data from {self.predict_f}")
        
         # don't catch Exceptions here, we want to stop
        json_collection = json.load(open(self.predict_f, "r"))        
        self.predict_documents = [Document(doc, self.taxo) for doc in json_collection]
        

    def print_taxo(self):
        for i, desc in self.taxo:
            print(f"{i} -> '{desc}'")
        
