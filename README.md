# Pre-Training Data Curation with LLMs

_Disclaimer: I threw this repo together extremely quickly as a feasibility demonstration of pre-training data curation with LLMs. It is *very* rough and is *definitely not* meant for use in production._

## About This Project

This project provides a mechanism for curating the pre-training data for an LLM, *with* an LLM. Curation takes two forms: *filtering* data points that shouldn't be included in the training data _at all_, and *revising* data points to better adhere to user-defined standards.

This project aims to *steer, influence, or limit* a base model's *overall behavior* in a completely flexible user-defined way. This approach was largely inspired by Anthropic's work on Constitutional AI, but instead of using LLMs to generate fine-tuning content based on user-defined principles, it uses an existing LLM to filter or revise *a new LLM's pre-training data*.

Critically, if a model doesn't have data displaying certain characteristics in its base dataset, it *should* be more difficult to trigger those characteristics in the resultant model than in one that was solely fine-tuned not to exhibit them. For example, it should be much more robust to jailbreaks, because rather than certain capabilities being inhibited by fine-tuning or reinforcement learning, the model simply doesn't have those capabilities in the first place.


## Running Data Cleaning

1. Create a `.env` file and inside of it instantiate env variable named OPENAI_API_KEY containing your OpenAI API key.
2. Download the pre-training dataset and place it in a `data/raw` folder in the repo's root directory
3. Run `python src/clean_data.py`
4. Cleaned data will be written in a `data/processed` folder

## Automated Tests

This repo uses `pytest` for testing. To run tests:

`pytest tests/filename.py`

Keep in mind that ~many of the tests will involve queries to an LLM, which will require adding your API key (see data cleaning instructions) incur any associated costs associated with the queries.


## Implementation Details

This method should be performed after pre-training data has been scraped, grouped, and had the standard non-LLM data curation steps performed on it for maximum efficiency.

Pre-training data is stored locally in a JSONL file. I used the webtext test set as an example of handling a large file line by line, but the specific format of the data is an easily changed implementation detail - as long as the training text is in discrete chunks that fit within the reviser LLM's context window, any format will serve.

Decisions about whether to filter or revise data are made by running each data point through an `Assessment` class. This class returns a probability estimate as an integer between 0 and 100 indicating how likely it is that the data point does not conform to the end user's criteria. In my example, this is done with an LLM that estimates the *deceptiveness* of content based few-shot examples, however it could also be done not only with other LLMs or with multi-step prompts, but it could also be combined or replaced with classifiers, regexes, or any other estimator.

The assessment's probability is compared to two user-defined thresholds: a `FILTER_THRESHOLD` and a `REVISE_THRESHOLD`, which determine whether a given datapoint should be filtered or revised, respectively. These thresholds should be tuned by the end-user for each use-case.

Similar to assessments, revisions are performed by a `Revision` class, which transforms any input string to a new string better conforming to the end user's criteria. In my example, this is done with an LLM that attempts to rewrite content to be less *deceptive*. However, like assessments, the internal workings of the revision can be customized as desired.
