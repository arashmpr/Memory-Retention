import utils

dataset_path = ""
train_path = ""
test_path = ""
val_path = ""
base_path = ""

class DatasetCreation():
  def __init__(self, glove_model, min_word_size=10000):

    self.vocabulary = set()
    self.positive_vocabulary = list()
    self.stop_words = set(utils.stopwords.words('english'))
    self.english_words = set(utils.words.words())
    self.min_word_size = min_word_size
    self.glove_model = glove_model

    with open('bias_lexicon.txt', 'r') as file:
      self.bias_words = file.read().splitlines()

    self.vad_df = pd.read_csv('NRC-VAD-Lexicon.txt', delimiter='\t', names=['Words', 'valence', 'arousal', 'dominance'])
    self.vad_words = set(self.vad_df['Words'].values)
    self.bias_words = set(self.bias_words)

    self.vad_bias_words = list(self.vad_words.union(self.bias_words))

    self.final_word_mapping = {}

    self.vad_bias_dict = {}
    for vad_bias in self.vad_bias_words:
      self.vad_bias_dict[vad_bias] = []

  def load_data(self):
    self.dataset = ds.load_dataset('lambada')
    self.trainset = self.dataset['train']['text']
    self.testset = self.dataset['test']['text']
    self.dataset = self.trainset + self.testset
    print("{} paragraphs.\n".format(len(self.dataset)))

  def create_paragraphs(self):
    times = 2
    time = 0
    result_cleaned_paragraphs = []
    cleaned_paragraphs = self.df['cleaned_paragraph'].values
    for paragraph in cleaned_paragraphs:
      time = 0
      res_paragraph = []
      sentences = nltk.sent_tokenize(paragraph)
      for sentence in sentences:
        tokens = sentence.split()
        if(len(res_paragraph) <= self.min_word_size):
          res_paragraph.extend(tokens)
        else:
          result_cleaned_paragraphs.append(' '.join(res_paragraph))
          if time < times:
            time += 1
            res_paragraph = []
            res_paragraph.extend(tokens)
            continue
          else:
            break
    return result_cleaned_paragraphs


  def create_dataframe(self):
    self.df = pd.DataFrame(self.dataset, columns=['raw_paragraph'])

  def is_meaningful(self, word):
    return word in self.english_words and word not in self.stop_words

  def clean_paragraph(self, paragraph):
    text = re.sub('\s+', ' ', paragraph.strip())
    text = text.lower()
    return text

  def create_vocabulary(self, paragraph):
    words = nltk.word_tokenize(paragraph)
    self.vocabulary.update(words)

  def has_negative_word(self, paragraph, positive_word):
    if positive_word in self.glove_model.key_to_index:
      similar_words = self.glove_model.most_similar(positive_word)
      for similar_word, _ in similar_words:
        if similar_word not in paragraph:
          return True
    return False

  def complete_vad_bias_dictionary(self, cleaned_paragraphs):
    all_len = len(self.vad_bias_dict.keys())
    for i, paragraphs in enumerate(cleaned_paragraphs):
      tokens = paragraphs.split()
      intersect = set(tokens).intersection(self.vad_bias_words)
      for vad_bias_word in list(intersect):
        self.vad_bias_dict[vad_bias_word].append(i)
    empty_keys = [key for key, value in self.vad_bias_dict.items() if len(value) == 0]
    for key in empty_keys:
      del self.vad_bias_dict[key]
    modified_len = len(self.vad_bias_dict.keys())
    print("{} words are in paragraphs out of {}".format(modified_len, all_len))

  def delete_index_from_all_dictionary(self, index):
    delete_these = []
    for key, value in self.vad_bias_dict.items():
      if index in value:
        delete_these.append(key)
    for key in delete_these:
      self.vad_bias_dict[key].remove(index)

    empty_keys = [key for key, value in self.vad_bias_dict.items() if len(value) == 0]
    for key in empty_keys:
      del self.vad_bias_dict[key]


  def assign_vad_bias(self):
    sorted_dict = dict(sorted(self.vad_bias_dict.items(), key=lambda item: len(item[1])))
    while(len(sorted_dict) != 0):
      key = list(sorted_dict.keys())[0]
      indexes = sorted_dict[key]
      index = random.choice(indexes)
      del self.vad_bias_dict[key]
      self.final_word_mapping[index] = key
      self.delete_index_from_all_dictionary(index)
      sorted_dict = dict(sorted(self.vad_bias_dict.items(), key=lambda item: len(item[1])))

  def random_positive_word(self, paragraph):
    searching = True
    dictionary = np.unique(paragraph.split())
    while(searching):
      positive_word = random.choice(dictionary)
      if self.is_meaningful(positive_word) and self.has_negative_word(paragraph, positive_word):
        return positive_word


  def fill_positive_words(self, paragraphs):
    self.positive_words = [0 for _ in range(len(paragraphs))]
    for i, paragraph in enumerate(paragraphs):
      if i in self.final_word_mapping.keys():
        self.positive_words[i] = self.final_word_mapping[i]
      else:
        self.positive_words[i] = self.random_positive_word(paragraph)

  def find_positive_words(self, cleaned_paragraphs):
    self.complete_vad_bias_dictionary(cleaned_paragraphs)
    self.assign_vad_bias()
    self.fill_positive_words(cleaned_paragraphs)

  def find_negative_words(self, paragraphs):
    self.negative_words = [0 for _ in range(len(self.positive_words))]
    for i, paragraph in tqdm(enumerate(paragraphs)):
      positive_word = self.positive_words[i]
      if positive_word in self.glove_model.key_to_index:
        similar_words = self.glove_model.most_similar(positive_word)
      else:
        continue
      for neg_word, _ in similar_words:
        if neg_word not in paragraph:
          self.negative_words[i] = neg_word
          break
    for i, negative_word in tqdm(enumerate(self.negative_words)):
      if negative_word == 0:
        positive_word = self.random_positive_word(paragraph)
        similar_words = self.glove_model.most_similar(positive_word)
        for neg_word, _ in similar_words:
          if neg_word not in paragraph:
            self.negative_words[i] = neg_word
            break
  def drop_duplicates(self):
    searching = True
    dup = self.df.drop_duplicates(subset=['Target Word'], keep='first')
    while (searching):
      pos_df = dup[dup['Label']==1]
      neg_df = dup[dup['Label']==0]
      print("we have {} positive unique words".format(len(pos_df)))
      print("we have {} negative unique words".format(len(neg_df)))
      if len(pos_df) == len(neg_df):
        break

      if len(neg_df) < len(pos_df):
        pos_df = pos_df[pos_df['Paragraph'].isin(neg_df['Paragraph'])]
      else:
        neg_df = neg_df[neg_df['Paragraph'].isin(pos_df['Paragraph'])]
      dup = pd.concat([pos_df, neg_df])
    self.df = dup

  def save(self):
    base_path = "/content/gdrive/MyDrive/Memory Retention Research/Datasets/Dataset_{}/Dataset_{}.csv".format(self.min_word_size, self.min_word_size)
    self.df.to_csv(base_path, index=False)

  def create(self):
    print("Creating..")
    self.df['cleaned_paragraph'] = self.df['raw_paragraph'].apply(lambda paragraph:self.clean_paragraph(paragraph))
    print("Paragraph Cleaned!")

    cleaned_paragraphs = self.create_paragraphs()
    print("We have {} paragraphs.".format(len(cleaned_paragraphs)))
    cleaned_paragraphs = np.unique(cleaned_paragraphs)
    print("Now we have {} unique paragraphs.".format(len(cleaned_paragraphs)))

    self.find_positive_words(cleaned_paragraphs)
    print("Positive words found. We have {} positive words.".format(len(self.positive_words)))
    self.find_negative_words(cleaned_paragraphs)
    print("Negative words found. We have {} negative words.".format(len(self.negative_words)))


    length = len(cleaned_paragraphs)
    label_1 = [1 for _ in range(length)]
    label_0 = [0 for _ in range(length)]

    cleaned_paragraphs = list(cleaned_paragraphs)
    self.positive_words = list(self.positive_words)
    self.negative_words = list(self.negative_words)

    cleaned_paragraphs.extend(cleaned_paragraphs)
    self.positive_words.extend(self.negative_words)
    label_1.extend(label_0)

    data = {
    'Paragraph': cleaned_paragraphs,
    'Target Word': self.positive_words,
    'Label': label_1
    }
    self.df = pd.DataFrame(data)
    self.drop_duplicates()

  def start(self):
    self.load_data()
    self.create_dataframe()
    self.create()
    print("Creation is DONE")
    self.save()
    print("Dataframe is SAVED")
    print()

dataset = DatasetCreation(glove_model, 10000)

dataset.start()

dataset.df.describe()