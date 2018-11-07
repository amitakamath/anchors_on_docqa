import argparse
from os.path import isfile

import re
import numpy as np
import tensorflow as tf

from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, PreserveParagraphs
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.utils import flatten_iterable, CachingResourceLoader
import pdb

"""
Script to run a model on user provided question/context document. 
This demonstrates how to use our document-pipeline on new input
"""

loader = None
sess = None
tokenizer = None
splitter = None 
selector = None
best_spans1, prediction, none_logit1, conf1 = None, None, None, None



def load_model(model_name):
   
    #parser = argparse.ArgumentParser(description="Run an ELMo model on user input")
    #parser.add_argument("model", help="Model directory")
    #parser.add_argument("question", help="Question to answer")
    #parser.add_argument("documents", help="List of text documents to answer the question with", nargs='+')
    #args = parser.parse_args()

    print("Preprocessing...")
    import pdb; #pdb.set_trace()
    # Load the model
    model_dir = ModelDir(model_name)
    model = model_dir.get_model()
    if not isinstance(model, ParagraphQuestionModel):
        raise ValueError("This script is built to work for ParagraphQuestionModel models only")
    global tokenizer
    tokenizer = NltkAndPunctTokenizer()

    global prediction
    global none_logit1
    global best_spans1
    global conf1

    global loader 
    loader = CachingResourceLoader()
    print('Loading word vectors...')
    model.set_input_spec(ParagraphAndQuestionSpec(batch_size=None), set([',']),
                       word_vec_loader=loader, allow_update=True)
    global sess
    print("Build tf graph")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))    
    with sess.as_default():
        # 8 means to limit the span to size 8 or less
        #best_spans, conf = model.get_prediction().get_best_span(8)
        global prediction, none_logit, best_spans, conf
        prediction = model.get_prediction()
        none_logit1 = prediction.none_logit  # This might have to become what it is plus [0]
        best_spans1, conf1 = prediction.get_best_span(8)
        
        #start_logits_tf = prediction.start_logits[0]
        #end_logits_tf = prediction.end_logits[0]
        #none_logit_tf = prediction.none_logit[0]

    model_dir.restore_checkpoint(sess)

    global splitter
    splitter = MergeParagraphs(400)

    global selector
    selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)

    # Read the documents
    """
    documents = []
    for doc in args.documents:
        if not isfile(doc):
            raise ValueError(doc + " does not exist")
        with open(doc, "r") as f:
            documents.append(f.read())
    print("Loaded %d documents" % len(documents))

    # Split documents into lists of paragraphs
    documents = [re.split("\s*\n\s*", doc) for doc in documents]
    """
    return model

def query(model, context, question):
    # Tokenize the input, the models expects data to be tokenized using `NltkAndPunctTokenizer`
    # Note the model expects case-sensitive input
    #model_dir = ModelDir('/u/scr/kamatha/document-qa/model-1029-021926')
    #tokenizer = NltkAndPunctTokenizer()
    question = tokenizer.tokenize_paragraph_flat(question)  # List of words
    # Now list of document->paragraph->sentence->word
    #documents = [[tokenizer.tokenize_paragraph(p) for p in doc] for doc in documents]
    documents = [[tokenizer.tokenize_paragraph(context)]]
    # Now group the document into paragraphs, this returns `ExtractedParagraph` objects
    # that additionally remember the start/end token of the paragraph within the source document
    #splitter = MergeParagraphs(400)
    # splitter = PreserveParagraphs() # Uncomment to use the natural paragraph grouping
    documents = [splitter.split(doc) for doc in documents]

    global selector

    # Now select the top paragraphs using a `ParagraphFilter`
    if len(documents) == 1:
        # Use TF-IDF to select top paragraphs from the document
        #selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
        context = selector.prune(question, documents[0])
    else:
        # Use a linear classifier to select top paragraphs among all the documents
        selector = ShallowOpenWebRanker(n_to_select=10)
        context = selector.prune(question, flatten_iterable(documents))

    #print("Select %d paragraph" % len(context))
    #pdb.set_trace()
    if model.preprocessor is not None:
        # Models are allowed to define an additional pre-processing step
        # This will turn the `ExtractedParagraph` objects back into simple lists of tokens
        context = [model.preprocessor.encode_text(question, x) for x in context]
    else:
        # Otherwise just use flattened text
        context = [flatten_iterable(x.text) for x in context]

    #print("Setting up model")
    # Tell the model the batch size (can be None) and vocab to expect, This will load the
    # needed word vectors and fix the batch size to use when building the graph / encoding the input
    voc = set(question)
    for txt in context:
        voc.update(txt)
    #model.set_input_spec(ParagraphAndQuestionSpec(batch_size=len(context)), voc)
    ###loader = CachingResourceLoader()
    ###model.set_input_spec(ParagraphAndQuestionSpec(batch_size=len(context)), voc, word_vec_loader=loader)
    #model.set_input_spec(ParagraphAndQuestionSpec(batch_size=len(context)), set([',']),
    #                     word_vec_loader=loader)

    # Now we build the actual tensorflow graph, `best_span` and `conf` are
    # tensors holding the predicted span (inclusive) and confidence scores for each
    # element in the input batch, confidence scores being the pre-softmax logit for the span
    #print("Build tf graph")

    #tf.reset_default_graph()
    #pdb.set_trace()
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # We need to use sess.as_default when working with the cuNND stuff, since we need an active
    # session to figure out the # of parameters needed for each layer. The cpu-compatible models don't need this.
    """
    with sess.as_default():
        # 8 means to limit the span to size 8 or less
        #best_spans, conf = model.get_prediction().get_best_span(8)
        prediction = model.get_prediction()
        none_logit = prediction.none_logit
        best_spans, conf = prediction.get_best_span(8)
   """
    # Loads the saved weights
    # model_dir.restore_checkpoint(sess)

    # Now the model is ready to run
    # The model takes input in the form of `ContextAndQuestion` objects, for example:
    data = [ParagraphAndQuestion(x, question, None, "user-question%d"%i)
            for i, x in enumerate(context)]

    #print("Starting run")
    # The model is run in two steps, first it "encodes" a batch of paragraph/context pairs
    # into numpy arrays, then we use `sess` to run the actual model get the predictions
    model.word_embed.update(loader, voc)
    encoded = model.encode(data, is_train=False)  # batch of `ContextAndQuestion` -> feed_dict

    global none_logit1
    global best_spans1
    global conf1
    best_spans, conf, none_logit = sess.run([best_spans1, conf1, none_logit1], feed_dict=encoded)  # feed_dict -> predictions
    #print(best_spans)
    #print(conf)
    #print(none_logit)

    best_para = np.argmax(conf)  # We get output for each paragraph, select the most-confident one to print
    #print("Best Paragraph: " + str(best_para))
    #print("Best span: " + str(best_spans[best_para]))
    #print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
    #print("Confidence: " + str(conf[best_para]))
    #print("No-Answer Confidence: " + str(none_logit[best_para]))
    answer = " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1])
    return answer

