
# ## 1. Loading the NIPS papers

#First, I explored the CSV file to determine what type of data I can use for the analysis and how it is structured. A research paper typically consists of a title, an abstract and the main text. Other data such as figures and tables were not extracted from the PDF files. Each paper discusses a novel technique or improvement. In this analysis, I focused on analyzing these papers with natural language processing methods.



# Importing modules
import pandas as pd

# Read datasets/papers.csv into papers
papers = pd.read_csv("datasets/papers.csv")

# Print out the first rows of papers
papers.head()


# ## 2. Preparing the data for analysis

# For the analysis of the papers, I am only interested in the text data associated with the paper as well as the year the paper was published in.
# I analyzed this text data using natural language processing.  Since the file contains some metadata such as id's and filenames, it is necessary to remove all the columns that do not contain useful text information.


# Remove the columns
papers.drop(columns=['id', 'event_type', 'pdf_name'], inplace=True)

# Print out the first rows of papers
papers.head()



# ## 3. Plotting how machine learning has evolved over time

# In order to understand how the machine learning field has recently exploded in popularity, we will begin by visualizing the number of publications per year.
# By looking at the number of published papers per year,  we can understand the extent of the machine learning 'revolution'! 



# Group the papers by year
groups = papers.groupby('year')

# Determine the size of each group
counts = groups.size()

# Visualise the counts as a bar plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
counts.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Papers per Group')
plt.title('Machine Learning Publications since 1987')



# ## 4. Preprocessing the text data

# Let's now analyze the titles of the different papers to identify machine learning trends. First, I performed some simple preprocessing on the titles in order to make them more amenable for analysis, use a regular expression to remove any punctuation in the title. Then performed lowercasing and then print the titles of the first rows before and after applying the modification.



# Load the regular expression library
import re

# Print the titles of the first rows 
print(papers['title'].head())

# Remove punctuation
papers['title_processed'] = papers['title'].map(lambda x: re.sub('[,\.!?]', '', x))

# Convert the titles to lowercase
papers['title_processed'] = papers['title_processed'].str.lower()

# Print the processed titles of the first rows 
papers.head()



# ## 5.  A word cloud to visualize the preprocessed text data

# In order to verify whether the preprocessing happened correctly, we can make a word cloud of the titles of the research papers. This will give us a visual representation of the most common words. Visualisation is key to understanding whether we are still on the right track! In addition, it allows us to verify whether we need additional preprocessing before further analyzing the text data.
#  Andreas Mueller's wordcloud library ("http://amueller.github.io/word_cloud/") is used here.


# Import the wordcloud library
import wordcloud

# Join the different processed titles together.
long_string = ' '.join(papers['title_processed'])

# Create a WordCloud object
wordcloud = wordcloud.WordCloud()

# Generate a word cloud
wordcloud.generate(long_string)
print(wordcloud.words_)

# Visualize the word cloud
wordcloud.to_image()



# ## 6.  Prepare the text for LDA analysis

# The main text analysis method that I used is latent Dirichlet allocation (LDA). LDA is able to perform topic detection on large document sets, determining what the main 'topics' are in a large unlabeled set of texts. 
# LDA does not work directly on text data. First, it is necessary to convert the documents into a simple vector representation. This representation will then be used by LDA to determine the topics. Each entry of a 'document vector' will correspond with the number of times a word occurred in the document. 


# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    print("total count ", total_counts)
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 

    plt.bar(x_pos, counts,align='center')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.title('10 most common words')
    plt.show()

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer()

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(papers['title_processed'])


# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)




# ## 7. Analysing trends with LDA

# Finally, the research titles will be analyzed using LDA. 



# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# number_topics defines the total number of topics in the LDA model.
# number_words is only for debugging purposes

number_topics = 10
number_words = 10

# Create and fit the LDA model
lda = LDA(n_components=number_topics)
lda.fit(count_data)

# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)



# ## 8. The future of machine learning
# Machine learning has become increasingly popular over the past years. The number of NIPS conference papers has risen exponentially, and people are continuously looking for ways on how they can incorporate machine learning into their products and services.

