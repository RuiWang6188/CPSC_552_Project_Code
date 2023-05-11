# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments

import csv
import json
import os

import datasets
from PIL import Image

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {50States10K},
author={Rui
},
year={2023}
}
"""

# You can copy an official description
_DESCRIPTION = """\
50States10K dataset
"""

_HOMEPAGE = "https://huggingface.co/datasets/Rui6188/50States10K"

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "Alabama": "Alabama",
    "Alaska": "Alaska",
    "Arizona": "Arizona",
    "Arkansas": "Arkansas",
    "California": "California",
    "Colorado": "Colorado",
    "Connecticut": "Connecticut",
    "Delaware": "Delaware",
    "Florida": "Florida",
    "Georgia": "Georgia",
    "Hawaii": "Hawaii",
    "Idaho": "Idaho",
    "Illinois": "Illinois",
    "Indiana": "Indiana",
    "Iowa": "Iowa",
    "Kansas": "Kansas",
    "Kentucky": "Kentucky",
    "Louisiana": "Louisiana",
    "Maine": "Maine",
    "Maryland": "Maryland",
    "Massachusetts": "Massachusetts",
    "Michigan": "Michigan",
    "Minnesota": "Minnesota",
    "Mississippi": "Mississippi",
    "Missouri": "Missouri",
    "Montana": "Montana",
    "Nebraska": "Nebraska",
    "Nevada": "Nevada",
    "New Hampshire": "New Hampshire",
    "New Jersey": "New Jersey",
    "New Mexico": "New Mexico",
    "New York": "New York",
    "North Carolina": "North Carolina",
    "North Dakota": "North Dakota",
    "Ohio": "Ohio",
    "Oklahoma": "Oklahoma",
    "Oregon": "Oregon",
    "Pennsylvania": "Pennsylvania",
    "Rhode Island": "Rhode Island",
    "South Carolina": "South Carolina",
    "South Dakota": "South Dakota",
    "Tennessee": "Tennessee",
    "Texas": "Texas",
    "Utah": "Utah",
    "Vermont": "Vermont",
    "Virginia": "Virginia",
    "Washington": "Washington",
    "West Virginia": "West Virginia",
    "Wisconsin": "Wisconsin",
    "Wyoming": "Wyoming",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Alabama", version=VERSION, description="images of Alamaba", data_dir="./Alabama"),
        datasets.BuilderConfig(name="Alaska", version=VERSION, description="images of Alaska", data_dir="./Alaska"),
        datasets.BuilderConfig(name="Arizona", version=VERSION, description="images of Arizona", data_dir="./Arizona"),
        datasets.BuilderConfig(name="Arkansas", version=VERSION, description="images of Arkansas", data_dir="./Arkansas"),
        datasets.BuilderConfig(name="California", version=VERSION, description="images of California", data_dir="./California"),
        datasets.BuilderConfig(name="Colorado", version=VERSION, description="images of Colorado", data_dir="./Colorado"),
        datasets.BuilderConfig(name="Connecticut", version=VERSION, description="images of Connecticut", data_dir="./Connecticut"),
        datasets.BuilderConfig(name="Delaware", version=VERSION, description="images of Delaware", data_dir="./Delaware"),
        datasets.BuilderConfig(name="Florida", version=VERSION, description="images of Florida", data_dir="./Florida"),
        datasets.BuilderConfig(name="Georgia", version=VERSION, description="images of Georgia", data_dir="./Georgia"),
        datasets.BuilderConfig(name="Hawaii", version=VERSION, description="images of Hawaii", data_dir="./Hawaii"),
        datasets.BuilderConfig(name="Idaho", version=VERSION, description="images of Idaho", data_dir="./Idaho"),
        datasets.BuilderConfig(name="Illinois", version=VERSION, description="images of Illinois", data_dir="./Illinois"),
        datasets.BuilderConfig(name="Indiana", version=VERSION, description="images of Indiana", data_dir="./Indiana"),
        datasets.BuilderConfig(name="Iowa", version=VERSION, description="images of Iowa", data_dir="./Iowa"),
        datasets.BuilderConfig(name="Kansas", version=VERSION, description="images of Kansas", data_dir="./Kansas"),
        datasets.BuilderConfig(name="Kentucky", version=VERSION, description="images of Kentucky", data_dir="./Kentucky"),
        datasets.BuilderConfig(name="Louisiana", version=VERSION, description="images of Louisiana", data_dir="./Louisiana"),
        datasets.BuilderConfig(name="Maine", version=VERSION, description="images of Maine", data_dir="./Maine"),
        datasets.BuilderConfig(name="Maryland", version=VERSION, description="images of Maryland", data_dir="./Maryland"),
        datasets.BuilderConfig(name="Massachusetts", version=VERSION, description="images of Massachusetts", data_dir="./Massachusetts"),
        datasets.BuilderConfig(name="Michigan", version=VERSION, description="images of Michigan", data_dir="./Michigan"),
        datasets.BuilderConfig(name="Minnesota", version=VERSION, description="images of Minnesota", data_dir="./Minnesota"),
        datasets.BuilderConfig(name="Mississippi", version=VERSION, description="images of Mississippi", data_dir="./Mississippi"),
        datasets.BuilderConfig(name="Missouri", version=VERSION, description="images of Missouri", data_dir="./Missouri"),
        datasets.BuilderConfig(name="Montana", version=VERSION, description="images of Montana", data_dir="./Montana"),
        datasets.BuilderConfig(name="Nebraska", version=VERSION, description="images of Nebraska", data_dir="./Nebraska"),
        datasets.BuilderConfig(name="Nevada", version=VERSION, description="images of Nevada", data_dir="./Nevada"),
        datasets.BuilderConfig(name="New Hampshire", version=VERSION, description="images of New Hampshire", data_dir="./New Hampshire"),
        datasets.BuilderConfig(name="New Jersey", version=VERSION, description="images of New Jersey", data_dir="./New Jersey"),
        datasets.BuilderConfig(name="New Mexico", version=VERSION, description="images of New Mexico", data_dir="./New Mexico"),
        datasets.BuilderConfig(name="New York", version=VERSION, description="images of New York", data_dir="./New York"),
        datasets.BuilderConfig(name="North Carolina", version=VERSION, description="images of North Carolina", data_dir="./North Carolina"),
        datasets.BuilderConfig(name="North Dakota", version=VERSION, description="images of North Dakota", data_dir="./North Dakota"),
        datasets.BuilderConfig(name="Ohio", version=VERSION, description="images of Ohio", data_dir="./Ohio"),
        datasets.BuilderConfig(name="Oklahoma", version=VERSION, description="images of Oklahoma", data_dir="./Oklahoma"),
        datasets.BuilderConfig(name="Oregon", version=VERSION, description="images of Oregon", data_dir="./Oregon"),
        datasets.BuilderConfig(name="Pennsylvania", version=VERSION, description="images of Pennsylvania", data_dir="./Pennsylvania"),
        datasets.BuilderConfig(name="Rhode Island", version=VERSION, description="images of Rhode Island", data_dir="./Rhode Island"),
        datasets.BuilderConfig(name="South Carolina", version=VERSION, description="images of South Carolina", data_dir="./South Carolina"),
        datasets.BuilderConfig(name="South Dakota", version=VERSION, description="images of South Dakota", data_dir="./South Dakota"),
        datasets.BuilderConfig(name="Tennessee", version=VERSION, description="images of Tennessee", data_dir="./Tennessee"),
        datasets.BuilderConfig(name="Texas", version=VERSION, description="images of Texas", data_dir="./Texas"),
        datasets.BuilderConfig(name="Utah", version=VERSION, description="images of Utah", data_dir="./Utah"),
        datasets.BuilderConfig(name="Vermont", version=VERSION, description="images of Vermont", data_dir="./Vermont"),
        datasets.BuilderConfig(name="Virginia", version=VERSION, description="images of Virginia", data_dir="./Virginia"),
        datasets.BuilderConfig(name="Washington", version=VERSION, description="images of Washington", data_dir="./Washington"),
        datasets.BuilderConfig(name="West Virginia", version=VERSION, description="images of West Virginia", data_dir="./West Virginia"),
        datasets.BuilderConfig(name="Wisconsin", version=VERSION, description="images of Wisconsin", data_dir="./Wisconsin"),
        datasets.BuilderConfig(name="Wyoming", version=VERSION, description="images of Wyoming", data_dir="./Wyoming"),
    ]
    

    DEFAULT_CONFIG_NAME = "Alabama"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        # if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option1": datasets.Value("string"),
        #             "answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        features = datasets.Features(
            {
                "image0": datasets.Image(),
                "image1": datasets.Image(),
                "image2": datasets.Image(),
                "image3": datasets.Image(),
                "label": datasets.Value(dtype='string', id=None),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]

        data_dir = dl_manager.download(urls)
        print(f"\ndata_dir: {data_dir}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir),
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # print(f"filepath: {filepath}")
        file_list = os.listdir(filepath)
        file_list = list(map(lambda x: os.path.join(filepath, x), filter(lambda x: x.endswith('.jpg'), file_list)))
        # print(f"len(file_list): {len(file_list)}")

        assert len(file_list) % 4 == 0, "image number cannot divided by 4"
        for i in range(0, len(file_list), 4):
            batch_files = file_list[i:i+4]
            yield i, {
                "image0": {"path": os.path.join(filepath, batch_files[0])},
                "image1": {"path": os.path.join(filepath, batch_files[1])},
                "image2": {"path": os.path.join(filepath, batch_files[2])},
                "image3": {"path": os.path.join(filepath, batch_files[3])},
                "label": filepath.split('/')[-1],
            }
