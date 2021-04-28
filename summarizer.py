import spacy
import en_core_web_trf
from collections import Counter
from heapq import nlargest

document1 ="""Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics."""
document2 = """Our Father who art in heaven, hallowed be thy name. Thy kingdom come. Thy will be done, on earth as it is in heaven. Give us this day our daily bread; and forgive us our trespasses, as we forgive those who trespass against us; and lead us not into temptation, but deliver us from evil
"""
document3 = "Nigeria has a varied landscape. The far south is defined by its tropical rainforest climate, where annual rainfall is 60 to 80 inches (1,500 to 2,000 mm) a year.[118] In the southeast stands the Obudu Plateau. Coastal plains are found in both the southwest and the southeast.[119] This forest zone's most southerly portion is defined as \"salt water swamp\", also known as a mangrove swamp because of the large amount of mangroves in the area. North of this is fresh water swamp, containing different vegetation from the salt water swamp, and north of that is rainforest.[120]\
    Climate map of Nigeria \
Nigeria's most expansive topographical region is that of the valleys of the Niger and Benue river valleys (which merge and form a Y-shape).[119] To the southwest of the Niger is \"rugged\" highland. To the southeast of the Benue are hills and mountains, which form the Mambilla Plateau, the highest plateau in Nigeria. This plateau extends through the border with Cameroon, where the montane land is part of the Bamenda Highlands of Cameroon. \
The area near the border with Cameroon close to the coast is rich rainforest and part of the Cross-Sanaga-Bioko coastal forests ecoregion, an important centre for biodiversity. It is habitat for the drill monkey, which is found in the wild only in this area and across the border in Cameroon. The areas surrounding Calabar, Cross River State, also in this forest, are believed to contain the world's largest diversity of butterflies. The area of southern Nigeria between the Niger and the Cross Rivers has lost most of its forest because of development and harvesting by increased population, with it being replaced by grassland (see Cross-Niger transition forests).\
Everything in between the far south and the far north is savannah (insignificant tree cover, with grasses and flowers located between trees). Rainfall is more limited, to between 500 and 1,500 millimetres (20 and 60 in) per year.[118] The savannah zone's three categories are Guinean forest-savanna mosaic, Sudan savannah, and Sahel savannah. Guinean forest-savanna mosaic is plains of tall grass interrupted by trees. Sudan savannah is similar but with shorter grasses and shorter trees. Sahel savannah consists of patches of grass and sand, found in the northeast.[120] In the Sahel region, rain is less than 500 millimetres (20 in) per year and the Sahara Desert is encroaching.[118] In the dry northeast corner of the country lies Lake Chad, which Nigeria shares with Niger, Chad and Cameroon."

nlp = en_core_web_trf.load()

def get_frequency_distribution(docx):

    '''Find weighted frequency via max frequency'''
    word_frequencies = Counter()
    for token in docx:
        if not token.is_stop and not token.is_punct:
            word_frequencies[token.text] += 1

    maximum_frequency = word_frequencies.most_common(1)[0][1]
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    return word_frequencies

def get_sentences_score(docx, word_frequencies):

    '''Sentence score and ranking of words in each sentence
    - Scoring every sentence based on number of non stopwords in our word frequency table
    '''
    sentence_scores = {}
    for sent in docx.sents:
        for word in sent:
            if word.lower_ in word_frequencies.keys() and len(sent.text.split(' ')) < 30:
                if sent in sentence_scores:
                    sentence_scores[sent] += word_frequencies[word.lower_]
                else:
                    sentence_scores[sent] = word_frequencies[word.lower_]
    return sentence_scores

def get_top_sentences(sentence_scores, no_of_summary_text=None):

    '''Finding Top N sentence with the largest score'''
    if no_of_summary_text:
        return nlargest(no_of_summary_text, sentence_scores, key=sentence_scores.get)
    return nlargest(7, sentence_scores, key=sentence_scores.get)


def text_summarizer(document):

    '''Summarize text'''
    docx = nlp(document)
    word_frequencies = get_frequency_distribution(docx)
    score = get_sentences_score(docx, word_frequencies)
    summarized_sentences = get_top_sentences(score)
    final_sentences = [word.text for word in summarized_sentences]

    return ''.join(final_sentences)

print('Length of Summarized Text:\t', len(text_summarizer(document3)))
print('Length of Initial Document:\t', len(document3))
print('*****SUMMARY:\n', text_summarizer(document3))
